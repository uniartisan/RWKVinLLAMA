from functools import partial
from src.model import Block,RWKV_Tmix_x070
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F


import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.accelerator import get_accelerator
from pytorch_lightning.strategies import DeepSpeedStrategy
# from adam_mini import Adam_mini
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
from train_functions import train_step, initialize_nccl_group, configure_optimizer, validation_step
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
        # Ensure hidden_states requires gradient
        hidden_states.requires_grad_(True)
        if self.args.grad_cp == 1:
            hidden_states = deepspeed.checkpointing.checkpoint(self.block, hidden_states)
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
    
class TimeMixWrapper(nn.Module):
    def __init__(self,args,layer_idx):
        super(TimeMixWrapper, self).__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.time_mixer = RWKV_Tmix_x070(args,layer_idx)
        
    def forward(self,
                hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
            position_embeddings,
            **kwargs,):
        args = self.args
        hidden_states.requires_grad_(True)
        if args.grad_cp == 1:
            x = deepspeed.checkpointing.checkpoint(self.time_mixer, hidden_states)
        else:
            x = self.time_mixer(hidden_states)
        return x,None,None


class HybridModel(pl.LightningModule):
    def __init__(self,transformer_model,rwkv_args,teacher_model = None,tokenizer=None):
        super(HybridModel, self).__init__()
        attn_num_heads = transformer_model.config.num_attention_heads
        attn_num_key_value_heads = transformer_model.config.num_key_value_heads
        assert attn_num_heads % attn_num_key_value_heads == 0
        n_share = attn_num_heads // attn_num_key_value_heads
        def init_block_params(rwkv_args,layer_idx,llama_layer):
            if rwkv_args.is_rwkv_att_only:
                decoder = llama_layer
                att = TimeMixWrapper(rwkv_args,layer_idx)
                att.time_mixer.receptance.weight.data = llama_layer.self_attn.q_proj.weight.data
                att.time_mixer.key.weight.data = llama_layer.self_attn.k_proj.weight.data.repeat(n_share, 1)
                att.time_mixer.value.weight.data = llama_layer.self_attn.v_proj.weight.data.repeat(n_share, 1)
                att.time_mixer.output.weight.data = llama_layer.self_attn.o_proj.weight.data
                del llama_layer.self_attn
                llama_layer.self_attn = att
                return decoder
            else:
                decoder = RWKVDecoderLayer(rwkv_args,layer_idx)
                decoder.block.att.receptance.weight.data = llama_layer.self_attn.q_proj.weight.data
                decoder.block.att.key.weight.data = llama_layer.self_attn.k_proj.weight.data.repeat(n_share, 1)
                decoder.block.att.value.weight.data = llama_layer.self_attn.v_proj.weight.data.repeat(n_share, 1)
                decoder.block.att.output.weight.data = llama_layer.self_attn.o_proj.weight.data
                if rwkv_args.is_llama_ffn:
                    decoder.block.ffn = llama_layer.mlp
                else:
                    decoder.block.ffn.c_fc.weight.data = llama_layer.mlp.up_proj.weight.data
                    decoder.block.ffn.c_proj.weight.data = llama_layer.mlp.down_proj.weight.data
                return decoder
        for layer_idx in range(transformer_model.config.num_hidden_layers):
            if layer_idx in rwkv_args.layers:
                decoder = init_block_params(rwkv_args,layer_idx,transformer_model.model.layers[layer_idx])
                former_decoder = transformer_model.model.layers[layer_idx]
                transformer_model.model.layers[layer_idx] = decoder
                del former_decoder
                
        self.model = transformer_model
        self.args = rwkv_args
        self.teacher_model = teacher_model
        #free the teacher model
        if self.teacher_model is not None:
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.eval()
        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            if 'pad_token_id' not in self.tokenizer.__dict__:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        torch.cuda.empty_cache()
        self.comm = None
        self.stream = None
        self.recv_buffer = None
        self.teacher_hidden_states_buffer = None
    def forward(
        self,
        input_ids,
        **kwargs,
    ):
        return self.model(input_ids, **kwargs)
    
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
        self.comm, self.stream, self.recv_buffer, self.teacher_hidden_states_buffer = initialize_nccl_group(self.args, self.model)
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
            



