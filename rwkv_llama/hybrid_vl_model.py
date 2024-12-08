from functools import partial
import os
RWKV_VERSION=os.environ.get('RWKV_VERSION','v7')
is_rwkv_7 = RWKV_VERSION == 'v7'
if is_rwkv_7 :
    from rwkv7.src.model import Block
else:
    from rwkv.src.model import Block
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
import deepspeed
import logging
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
class RWKVVLDecoderLayer(nn.Module):
    def __init__(
        self,
        args,
        layer_idx: int
    ):
        super(RWKVVLDecoderLayer, self).__init__()
        self.block = Block(args,layer_idx)
        self.layer_idx = layer_idx
        self.args = args

    def forward(self, hidden_states: torch.Tensor, inference_params=None, *args, **kwargs):
        hidden_states.requires_grad_(True)
        if is_rwkv_7:
            #if we don't have v_first in kwargs, we create an empty v_first tensor
            if 'v_first' not in kwargs:
                kwargs['v_first'] = torch.empty_like(hidden_states)
            v_first = kwargs['v_first']
        if self.args.grad_cp == 1:
            if is_rwkv_7:
                hidden_states,v_first = deepspeed.checkpointing.checkpoint(self.block, hidden_states, v_first)
                kwargs['v_first'] = v_first
            else:
                hidden_states = deepspeed.checkpointing.checkpoint(self.block, hidden_states)
        else:
            if is_rwkv_7:
                hidden_states,v_first = self.block(hidden_states, v_first)
            else:
                hidden_states = self.block(hidden_states)

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
    
def replace_llama_layers(transformer_model, rwkv_args):
    attn_num_heads = transformer_model.config.num_attention_heads
    attn_num_key_value_heads = transformer_model.config.num_key_value_heads
    assert attn_num_heads % attn_num_key_value_heads == 0
    n_share = attn_num_heads // attn_num_key_value_heads

    # 替换层的逻辑保持不变
    for layer_idx in range(transformer_model.config.num_hidden_layers):
        if layer_idx in rwkv_args.layers:
            decoder = RWKVVLDecoderLayer(rwkv_args, layer_idx)
            llama_layer = transformer_model.model.layers[layer_idx]
            
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

    return transformer_model

class HybridModel(nn.Module):
    def __init__(self, transformer_model, rwkv_args, tokenizer=None):
        super(HybridModel, self).__init__()
        attn_num_heads = transformer_model.config.num_attention_heads
        attn_num_key_value_heads = transformer_model.config.num_key_value_heads
        assert attn_num_heads % attn_num_key_value_heads == 0
        n_share = attn_num_heads // attn_num_key_value_heads

        # 替换层的逻辑保持不变
        for layer_idx in range(transformer_model.config.num_hidden_layers):
            if layer_idx in rwkv_args.layers:
                decoder = RWKVVLDecoderLayer(rwkv_args, layer_idx)
                llama_layer = transformer_model.model.layers[layer_idx]
                
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


        
    def forward(
        self,
        input_ids,
        **kwargs,
    ):
        return self.model(input_ids, **kwargs)
    
