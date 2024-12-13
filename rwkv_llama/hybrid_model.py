from functools import partial
import os
from typing import Optional, Tuple
RWKV_VERSION=os.environ.get('RWKV_VERSION','v7')
is_rwkv_7 = RWKV_VERSION == 'v7'
if is_rwkv_7 :
    from rwkv7.src.model import Block
    from rwkv7.src.model import RWKV_Tmix_x070 as TimeMixer
else:
    from rwkv.src.model import Block
    from rwkv.src.model import RWKV_Tmix_x060 as TimeMixer
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F


import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.accelerator import get_accelerator
from pytorch_lightning.strategies import DeepSpeedStrategy
# from adam_mini import Adam_mini
from transformers import AutoModelForCausalLM
import cupy as cp
from cupy.cuda import nccl
import logging
# from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
import os
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
from train_functions import train_step,  configure_optimizer, validation_step,initialize_nccl_client
v_first = None
class RWKVDecoderLayer(nn.Module):
    def __init__(
        self,
        args,
        layer_idx: int
    ):
        super(RWKVDecoderLayer, self).__init__()
        self.block = Block(args,layer_idx)
        self.layer_idx = layer_idx
        self.args = args

    def forward(self, hidden_states: torch.Tensor, inference_params=None, *args, **kwargs):
        hidden_states.requires_grad_(True)
        if is_rwkv_7:
            #if we don't have v_first in kwargs, we create an empty v_first tensor
            global v_first
            if v_first is None:
                v_first = torch.empty_like(hidden_states)
            #     print(f'empty v_first in layer {self.layer_idx}')
            # else:
            #     print(f'reuse v_first in layer {self.layer_idx}')
        if self.args.grad_cp == 1:
            if is_rwkv_7:
                hidden_states,v_first = deepspeed.checkpointing.checkpoint(self.block, hidden_states, v_first)
            else:
                hidden_states = deepspeed.checkpointing.checkpoint(self.block, hidden_states)
        else:
            if is_rwkv_7:
                hidden_states,v_first = self.block(hidden_states, v_first)
            else:
                hidden_states = self.block(hidden_states)
        # hidden_states = self.block(hidden_states)
        # logging.info(f'forward in {self.layer_idx}')
        # so here is just to be compatible with Transformer

        past_key_value = kwargs.get("past_key_value", None)

        if past_key_value is not None:
            dummy_keys = torch.ones(
                1, 1, hidden_states.size(1), 1, device=hidden_states.device, dtype=hidden_states.dtype
            )
            dummy_values = torch.ones(
                1, 1, hidden_states.size(1), 1, device=hidden_states.device, dtype=hidden_states.dtype
            )
            # Update kv cache with dummy values
            past_key_value.update(dummy_keys, dummy_values, self.layer_idx)

        return (hidden_states, None, past_key_value)
    



class AttentionWrapper(nn.Module):
    
    def __init__(self,teacher_attn,student_attn,args):
        super(AttentionWrapper, self).__init__()
        self.args = args
        if teacher_attn is not None:
            # 创建一个新的相同类型的 attention 模块
            self.teacher_attn = type(teacher_attn)(
                config=teacher_attn.config
            )
            # 复制状态字典
            self.teacher_attn.load_state_dict(teacher_attn.state_dict())
            # 确保在 CPU 上
            self.teacher_attn = self.teacher_attn.to('cpu')
            # 冻结参数
            for param in self.teacher_attn.parameters():
                param.requires_grad = False
            self.add_module("teacher_attn", self.teacher_attn)
            del teacher_attn
        else:
            self.teacher_attn = None
        self.student_attn = student_attn
        self.student_attn.requires_grad_(True)
        self.add_module("student_attn", self.student_attn)
    
    def forward(self, 
        # hidden_states: torch.Tensor,
        # attention_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        # past_key_value: Optional[Cache] = None,
        # output_attentions: bool = False,
        # use_cache: bool = False,
        # cache_position: Optional[torch.LongTensor] = None,
        # position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        kwargs['output_attentions'] = False

        # NOTE - instead of returning attentions here we return a special attention loss
        hidden_states = kwargs['hidden_states']
        past_key_value = kwargs.get("past_key_value", None)
        hidden_states = hidden_states.requires_grad_(True)
        if is_rwkv_7:
            #if we don't have v_first in kwargs, we create an empty v_first tensor
            global v_first
            if v_first is None:
                v_first = torch.empty_like(hidden_states)
        if self.args.grad_cp == 1:
            if is_rwkv_7:
                student_hidden_states,v_first = deepspeed.checkpointing.checkpoint(self.student_attn, hidden_states, v_first)
            else:
                student_hidden_states = deepspeed.checkpointing.checkpoint(self.student_attn, hidden_states)
        else:
            if is_rwkv_7:
                student_hidden_states,v_first = self.student_attn(hidden_states, v_first)
            else:
                student_hidden_states = self.student_attn(hidden_states)
        if self.args.stage == 2:
            return (student_hidden_states, None, past_key_value)
        # student_outputs = self.student_attn(hidden_states)
        with torch.no_grad():
            teacher_outputs = self.teacher_attn(*args, **kwargs)
        # special attention loss is the vector norm of the difference between the student and teacher attn outputs
        # student_hidden_states = student_outputs[0]
        teacher_hidden_states = teacher_outputs[0]
        special_attn_loss = torch.linalg.vector_norm(teacher_hidden_states - student_hidden_states, dim=-1).mean() * (teacher_hidden_states[0].size(-1) ** -0.5)
        return (teacher_outputs[0], special_attn_loss, ) + teacher_outputs[2:]

class HybridModel(pl.LightningModule):
    def __init__(self, transformer_model, rwkv_args, tokenizer=None):
        super(HybridModel, self).__init__()
        attn_num_heads = transformer_model.config.num_attention_heads
        attn_num_key_value_heads = transformer_model.config.num_key_value_heads
        assert attn_num_heads % attn_num_key_value_heads == 0
        n_share = attn_num_heads // attn_num_key_value_heads
        stage = rwkv_args.stage
        if stage == 1:
            #Freeze the model
            transformer_model.requires_grad_(False)
        else:
            #Unfreeze the model
            transformer_model.requires_grad_(True)
            if transformer_model.config.tie_word_embeddings:
                # copy untied embeddings
                transformer_model.get_output_embeddings().weight = nn.Parameter(transformer_model.get_input_embeddings().weight.clone())
                # untie the embeddings in the config, too
                transformer_model.tie_word_embeddings = False
        # 替换层的逻辑保持不变
        for layer_idx in range(transformer_model.config.num_hidden_layers):
            if layer_idx in rwkv_args.layers:
                if not rwkv_args.is_rwkv_att_only:
                    decoder = RWKVDecoderLayer(rwkv_args, layer_idx)
                    llama_layer = transformer_model.model.layers[layer_idx]
                    if rwkv_args.init_with_llama:
                        print(f'init parameters with llama in layer {layer_idx}')
                        decoder.block.att.receptance.weight.data = llama_layer.self_attn.q_proj.weight.data
                        decoder.block.att.key.weight.data = llama_layer.self_attn.k_proj.weight.data.repeat(n_share, 1)
                        decoder.block.att.value.weight.data = llama_layer.self_attn.v_proj.weight.data.repeat(n_share, 1)
                        decoder.block.att.output.weight.data = llama_layer.self_attn.o_proj.weight.data
                    
                    if rwkv_args.is_llama_ffn:
                        decoder.block.ffn = llama_layer.mlp
                    else:
                        decoder.block.ffn.c_fc.weight.data = llama_layer.mlp.up_proj.weight.data
                        decoder.block.ffn.c_proj.weight.data = llama_layer.mlp.down_proj.weight.data
                    
                    transformer_model.model.layers[layer_idx] = decoder
                    del llama_layer
                else:
                    #Only replace the attention layer with TimeMixer
                    student_attn = TimeMixer(rwkv_args, layer_idx)
                    llama_layer = transformer_model.model.layers[layer_idx]
                    # if stage == 1:
                    #     teacher_attn = llama_layer.self_attn
                    # else:
                    #     teacher_attn = None
                    #Remove the teacher_attn out of the model which makes
                    #deepspeed can initialize the model easily
                    attn_wrapper = AttentionWrapper(None,student_attn,rwkv_args)
                    llama_layer.self_attn = attn_wrapper
                    import gc
                    gc.collect()
        self.model = transformer_model
        self.add_module("model", self.model)
        self.args = rwkv_args
        self.teacher_model = None  # 初始化为None，后续再设置
        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            if 'pad_token_id' not in self.tokenizer.__dict__:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        torch.cuda.empty_cache()
        self.client = None

    def set_teacher_model(self, teacher_model):
        """设置teacher model的方法"""
        self.teacher_model = teacher_model
        self.add_module('teacher_model', self.teacher_model)
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
        
    def forward(
        self,
        input_ids,
        **kwargs,
    ):
        global v_first
        v_first = None
        ret = self.model(input_ids, **kwargs)
        v_first = None
        return ret
    
    def configure_optimizers(self):
        return configure_optimizer(self, self.args)
    
    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False
    def on_fit_start(self):
        self.client = initialize_nccl_client(self.args)
    def validation_step(self, batch, batch_idx):
        result = validation_step(self, batch, self.args, self.teacher_model, self.tokenizer)
        self.log_dict(result, prog_bar=True)
        return result
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # 在每个训练批次结束时清空缓存
        try:
            get_accelerator().empty_cache()
        except AttributeError:
            # 如果get_accelerator()不可用,尝试使用torch.cuda
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"无法清空缓存: {e}")
    def training_step(self, batch, batch_idx):
        loss, teacher_loss, kl_loss, student_cross_entropy_loss = train_step(self, batch, self.args, self.teacher_model, self.tokenizer)
        return {"loss": loss, "teacher_loss": teacher_loss, "kl_loss": kl_loss, "student_cross_entropy_loss": student_cross_entropy_loss}
            



