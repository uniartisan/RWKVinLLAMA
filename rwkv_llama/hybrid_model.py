from src.model import RWKV_Tmix_x060, RWKV_CMix_x060,Block
import torch
import pytorch_lightning as pl
import torch.nn as nn


import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from pytorch_lightning.strategies import DeepSpeedStrategy
from adam_mini import Adam_mini



class RWVVDecoderLayer(nn.Module):
    def __init__(
        self,
        args,
        layer_idx: int
    ):
        super(RWVVDecoderLayer, self).__init__()
        self.block = Block(args,layer_idx)
        self.layer_idx = layer_idx

    def forward(self, hidden_states: torch.Tensor, inference_params=None, *args, **kwargs):
        hidden_states = self.block(hidden_states)
        # print(f'forward in {self.layer_idx}')
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

class HybridModel(pl.LightningModule):
    def __init__(self,transformer_model,rwkv_args):
        super(HybridModel, self).__init__()
        self.emb = transformer_model.get_input_embeddings()
        attn_num_heads = transformer_model.config.num_attention_heads
        attn_num_key_value_heads = transformer_model.config.num_key_value_heads
        assert attn_num_heads % attn_num_key_value_heads == 0
        n_share = attn_num_heads // attn_num_key_value_heads
        def init_block_params(rwkv_args,layer_idx,llama_layer):
            decoder = RWVVDecoderLayer(rwkv_args,layer_idx)
            decoder.block.att.receptance.weight.data = llama_layer.self_attn.q_proj.weight.data
            decoder.block.att.key.weight.data = llama_layer.self_attn.k_proj.weight.data.repeat(n_share, 1)
            decoder.block.att.value.weight.data = llama_layer.self_attn.v_proj.weight.data.repeat(n_share, 1)
            decoder.block.att.output.weight.data = llama_layer.self_attn.o_proj.weight.data
            decoder.block.ffn.key.weight.data = llama_layer.mlp.up_proj.weight.data
            decoder.block.ffn.value.weight.data = llama_layer.mlp.down_proj.weight.data
            return decoder
        for layer_idx in range(transformer_model.config.num_hidden_layers):
            if layer_idx in rwkv_args.layers:
                rwkv_encoder = init_block_params(rwkv_args,layer_idx,transformer_model.model.layers[layer_idx])
                transformer_model.model.layers[layer_idx] = rwkv_encoder
        self.model = transformer_model
        self.args = rwkv_args
    
    def forward(
        self,
        input_ids,
        inference_params=None,
        **kwargs,
    ):
        return self.model(input_ids, **kwargs)
    
    def configure_optimizers(self):
        args = self.args
        
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if (("_w1" in n) or ("_w2" in n)) and (args.layerwise_lr > 0):
                lr_1x.add(n)
            elif (("time_mix" in n) or ("time_maa" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif (("time_decay" in n) or ("time_daaaa" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_3x.add(n)
                else:
                    lr_2x.add(n)
            elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif ("time_first" in n) and (args.layerwise_lr > 0):
                lr_3x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))
        # print('decay', lr_decay)
        # print('1x', lr_1x)
        # print('2x', lr_2x)
        # print('3x', lr_3x)
        param_dict = {n: p for n, p in self.named_parameters()}
        
        if args.layerwise_lr > 0:
            if args.my_pile_stage == 2:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 2e-3 / args.lr_init},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 3e-3 / args.lr_init},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]

        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
            if args.optim=='adam_mini':
                return Adam_mini(self, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, weight_decay=0, model_sharding=True, n_feature=args.n_embd, n_head=args.n_embd//64, lora_r=8)
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
        else:
            if args.optim=='adam_mini':
                return Adam_mini(self, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, weight_decay=0, model_sharding=True, n_feature=args.n_embd, n_head=args.n_embd//64, lora_r=8)
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
        # return ZeroOneAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False, cuda_aware=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def training_step(self, batch, batch_idx):
        args = self.args