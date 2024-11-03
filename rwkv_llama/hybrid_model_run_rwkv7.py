import sys
import os
def setup_env():
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path.append(parent_dir)
    rwkv_path = os.path.join(parent_dir, 'rwkv')
    sys.path.append(rwkv_path)
    rwkv_llama_path = os.path.join(parent_dir, 'rwkv_llama')
    sys.path.append(rwkv_llama_path)
    # print(f'add path: {rwkv_path} to sys.path')
    print(f'add path: {rwkv_llama_path} to sys.path')
    os.environ['CUDA_HOME'] = '/usr/local/cuda-12.1'
    os.environ['RWKV_JIT_ON'] = '0'
    os.environ['RWKV_T_MAX'] = '4096'
    os.environ['RWKV_FLOAT_MODE'] = 'bf16'
    os.environ['RWKV_HEAD_SIZE_A'] = '64'
    os.environ['RWKV_T_MAX'] = '4096'
    os.environ["RWKV_MY_TESTING"]='x060'
    os.environ['RWKV_CTXLEN'] = '4096'
    os.environ['WKV'] = 'fla'
    os.environ["RWKV_TRAIN_TYPE"] = ''
setup_env()
from einops import rearrange
# from fla.ops.rwkv6 import chunk_rwkv6,fused_recurrent_rwkv6
import math
from fla.ops.rwkv7.fused_recurrent import fused_recurrent_rwkv7
def RUN_CUDA_RWKV7_STATE(B, T, C, H, r, k, v, w, a, b,s):
    dtype = r.dtype
    r = rearrange(r, 'b l (h d) -> b h l d', h = H)
    k = rearrange(k, 'b l (h d) -> b h l d', h = H)
    v = rearrange(v, 'b l (h d) -> b h l d', h = H)
    w = rearrange(w, 'b l (h d) -> b h l d', h = H)
    o, state = fused_recurrent_rwkv7(r, k, v, w, a,b, scale=1., initial_state=s, output_final_state=True,training=False)
    x = rearrange(o, 'b h l d -> b l (h d)')
    return x.to(dtype), state.to(dtype)
import torch
from utilities import TimeMixState, ChannelMixState, BlockState
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple
import logging
from transformers.cache_utils import Cache,DynamicCache
# from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
import os
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

class RWKV_Tmix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.n_embd = args.n_embd

        self.head_size = args.head_size_a        
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # initialization comes from fitting my RWKV-6 7B runs
            # merging r&g w&a to save params
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, 0.6 * ratio_1_to_almost0 ** 0.9))
            self.time_maa_rg = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.time_maa_wa = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))

            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -7 + 5 * (n / (args.dim_att - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att) + 0.5) # !!! 0.5 comes from F.softplus !!!

            self.time_faaaa = nn.Parameter(torch.zeros(1,1,self.n_head,self.head_size))
            self.time_aaaaa = nn.Parameter(torch.zeros(1,1,args.dim_att))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            D_MIX_LORA = 32
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*4))
            self.time_maa_w2 = nn.Parameter(ortho_init(torch.zeros(4, D_MIX_LORA, args.n_embd), 0.1))

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, args.dim_att), 0.1))

            D_AAA_LORA = 16
            self.time_aaa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_AAA_LORA))
            self.time_aaa_w2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, args.dim_att), 0.1))

            D_KKK_LORA = 16
            self.time_kkk_w1 = nn.Parameter(torch.zeros(args.n_embd, D_KKK_LORA))
            self.time_kkk_w2 = nn.Parameter(ortho_init(torch.zeros(D_KKK_LORA, args.dim_att), 0.1))

            D_GATE_LORA = 128
            self.gate_w1 = nn.Parameter(ortho_init(torch.zeros(args.n_embd, D_GATE_LORA), 0.1))
            self.gate_w2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, args.dim_att), 0.1))

            D_MA_LORA = 16
            self.ma_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MA_LORA))
            self.ma_w2 = nn.Parameter(ortho_init(torch.zeros(D_MA_LORA, args.dim_att), 0.1))
            self.time_misc_a = nn.Parameter(torch.zeros(1,1,args.n_embd))
            D_MK_LORA = 16
            self.mk_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MK_LORA))
            self.mk_w2 = nn.Parameter(ortho_init(torch.zeros(D_MK_LORA, args.dim_att), 0.1))
            self.time_misc_k = nn.Parameter(torch.zeros(1,1,args.n_embd))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
            self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=64e-5)

            self.receptance.weight.data.uniform_(-0.5/(self.n_embd**0.5), 0.5/(self.n_embd**0.5))
            self.key.weight.data.uniform_(-0.05/(self.n_embd**0.5), 0.05/(self.n_embd**0.5))
            self.value.weight.data.uniform_(-0.5/(self.n_embd**0.5), 0.5/(self.n_embd**0.5))
            self.output.weight.data.zero_()

    def forward(self, x, last_state: TimeMixState):
        shift_state = last_state.shift_state
        B, T, C = x.size()
        H = self.n_head
        if shift_state is not None:
            xx = torch.concat((shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x
        else:
            xx = self.time_shift(x) - x
        lx = x[:, -1]
        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 4, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(4, B, T, -1)
        mrg, mwa, mk, mv = xxx.unbind(dim=0)

        xrg = x + xx * (self.time_maa_rg + mrg)
        xwa = x + xx * (self.time_maa_wa + mwa)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)

        r = self.receptance(xrg)
        w = -F.softplus(-(self.time_decay + torch.tanh(xwa @ self.time_decay_w1) @ self.time_decay_w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        g = torch.tanh(xrg @ self.gate_w1) @ self.gate_w2

        kk = k + torch.tanh(xk @ self.time_kkk_w1) @ self.time_kkk_w2
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        a = torch.sigmoid( self.time_aaaaa + (xwa @ self.time_aaa_w1) @ self.time_aaa_w2 )

        ma = torch.sigmoid(self.time_misc_a + (xwa @ self.ma_w1) @ self.ma_w2)
        k = k * ma + k*a * (1 - ma)
        mk = torch.sigmoid(self.time_misc_k + (xk @ self.mk_w1) @ self.mk_w2)
        k = k * torch.clamp(w*mk, max=0).exp()
        wkv_state = last_state.wkv_state
        x , wkv_state= RUN_CUDA_RWKV7_STATE(B, T, C, H,r.bfloat16(), k.bfloat16(), v.bfloat16(), w.bfloat16(), -kk.bfloat16(), (kk*a).bfloat16(),s=wkv_state)
        
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.time_faaaa).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x,TimeMixState(lx,wkv_state)


class RWKV_Tmix_x070_Wrapper(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_idx = layer_id
        self.time_mixer = RWKV_Tmix_x070(args, layer_id)

    def forward(self, 
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
            position_embeddings,
            **kwargs):
        x = hidden_states
        args = self.args
        B, T, C = x.size()
        if past_key_value is not None:
            if len(past_key_value) <= self.layer_idx:
                last_state = None
            else:
                last_state = past_key_value[self.layer_idx][0]
        if last_state is None:
            H =  args.dim_att // args.head_size_a
            device = x.device
            dtype = x.dtype
            wkv_states = torch.empty((B, H, C//H, C//H),
                                 device=device,
                                 dtype=dtype)
            token_shift = torch.empty((B,C),
                                 device=device,
                                 dtype=dtype)
            wkv_states[:] = 0
            token_shift[:] = 0
            time_state = TimeMixState(token_shift, wkv_states)
            # print(wkv_states)
            channel_state = None
            last_state = BlockState(time_state,channel_state)
        x,states= self.time_mixer(x,last_state.time_mix_state)
        last_state.time_mix_state = states
        if past_key_value is not None:
            keys = T
            values = last_state
            past_key_value.update(keys, values, self.layer_idx)
        return x,None,past_key_value    

class RWKV_CMix_x060_infctx(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    def forward(self, x, last_state: ChannelMixState):
        if last_state.shift_state is not None:
            xx = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x
        else:
            xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv, ChannelMixState(x[:, -1])
    
    
class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)

        self.ffn = RWKV_CMix_x060_infctx(args, layer_id)
        
    def forward(self, x, last_state: BlockState = None, x_emb=None):
        args = self.args
        B, T, C = x.size()
        if self.layer_id == 0:
            x = self.ln0(x)
        if last_state is None:
            H =  args.dim_att // args.head_size_a
            device = x.device
            dtype = x.dtype
            wkv_states = torch.empty((B, H, C//H, C//H),
                                 device=device,
                                 dtype=dtype)
            shift_states = torch.empty((2,B,C),
                                 device=device,
                                 dtype=dtype)
            wkv_states[:] = 0
            shift_states[:] = 0
            time_state = TimeMixState(shift_states[0], wkv_states)
            # print(wkv_states)
            channel_state = ChannelMixState(shift_states[1])
            last_state = BlockState(time_state,channel_state)
        if self.layer_id == 0 and args.pre_ffn > 0:
            x = x + self.ffnPre(self.ln1(x))
        else:
            att_out, att_state = self.att(self.ln1(x), last_state.time_mix_state)
            x = x + att_out
        if args.is_llama_ffn:
            ffn_out = self.ffn(self.ln2(x))
            fnn_state = None
        else:
            ffn_out, fnn_state = self.ffn(self.ln2(x), last_state.channel_mix_state)
        x = x + ffn_out
        last_state.time_mix_state = att_state
        last_state.channel_mix_state = fnn_state
        return x, last_state
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

    def forward(self, 
                hidden_states: torch.Tensor, 
                past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False, 
        output_attentions: Optional[bool] = False, 
        *args, 
        **kwargs):
        # Ensure hidden_states requires gradient
        _,T,_ = hidden_states.shape
        if past_key_value is not None:
            if len(past_key_value) <= self.layer_idx:
                last_state = None
            else:
                last_state = past_key_value[self.layer_idx][0]
        hidden_states,states= self.block(hidden_states,last_state)
        # hidden_states = self.block(hidden_states)
        # logging.info(f'forward in {self.layer_idx}')
        # so here is just to be compatible with Transformer

        # past_key_value = kwargs.get("past_key_value", None)

        if past_key_value is not None:
            keys = T
            values = states
            past_key_value.update(keys, values, self.layer_idx)
        outputs = (hidden_states,)
        if output_attentions :
            outputs += (None,)
        if use_cache:
            outputs += (past_key_value,)
        return outputs

class HybridModel(nn.Module):
    def __init__(self,transformer_model,rwkv_args):
        super(HybridModel, self).__init__()
        attn_num_heads = transformer_model.config.num_attention_heads
        attn_num_key_value_heads = transformer_model.config.num_key_value_heads
        assert attn_num_heads % attn_num_key_value_heads == 0
        n_share = attn_num_heads // attn_num_key_value_heads
        def init_block_params(rwkv_args,layer_idx,llama_layer):
            if rwkv_args.is_rwkv_att_only:
                decoder = llama_layer
                att = RWKV_Tmix_x070_Wrapper(rwkv_args,layer_idx)
                # att.time_mixer.receptance.weight.data = llama_layer.self_attn.q_proj.weight.data
                # att.time_mixer.key.weight.data = llama_layer.self_attn.k_proj.weight.data.repeat(n_share, 1)
                # att.time_mixer.value.weight.data = llama_layer.self_attn.v_proj.weight.data.repeat(n_share, 1)
                # att.time_mixer.output.weight.data = llama_layer.self_attn.o_proj.weight.data
                llama_layer.self_attn = att
                return decoder
            else:
                decoder = RWKVDecoderLayer(rwkv_args,layer_idx)
                # decoder.block.att.receptance.weight.data = llama_layer.self_attn.q_proj.weight.data
                # decoder.block.att.key.weight.data = llama_layer.self_attn.k_proj.weight.data.repeat(n_share, 1)
                # decoder.block.att.value.weight.data = llama_layer.self_attn.v_proj.weight.data.repeat(n_share, 1)
                # decoder.block.att.output.weight.data = llama_layer.self_attn.o_proj.weight.data
                if rwkv_args.is_llama_ffn:
                    decoder.block.ffn = llama_layer.mlp
                return decoder
            # decoder = RWKVDecoderLayer(rwkv_args,layer_idx)
            # if rwkv_args.is_llama_ffn:
            #     decoder.block.ffn = llama_layer.mlp
            # decoder.block.att.receptance.weight.data = llama_layer.self_attn.q_proj.weight.data
            # decoder.block.att.key.weight.data = llama_layer.self_attn.k_proj.weight.data.repeat(n_share, 1)
            # decoder.block.att.value.weight.data = llama_layer.self_attn.v_proj.weight.data.repeat(n_share, 1)
            # decoder.block.att.output.weight.data = llama_layer.self_attn.o_proj.weight.data
            # decoder.block.ffn.key.weight.data = llama_layer.mlp.up_proj.weight.data
            # decoder.block.ffn.value.weight.data = llama_layer.mlp.down_proj.weight.data
            return decoder
        for layer_idx in range(transformer_model.config.num_hidden_layers):
            if layer_idx in rwkv_args.layers:
                rwkv_encoder = init_block_params(rwkv_args,layer_idx,transformer_model.model.layers[layer_idx])
                old_layer = transformer_model.model.layers[layer_idx]
                transformer_model.model.layers[layer_idx] = rwkv_encoder
                del old_layer
        self.model = transformer_model
        self.args = rwkv_args
    def forward(
        self,
        input_ids,
        inference_params=None,
        **kwargs,
    ):
        return self.model(input_ids, **kwargs)
    def load_ckpt(self, ckpt_file):
        print(f'loading ckpt from {ckpt_file}')
        info = self.load_state_dict(torch.load(ckpt_file,weights_only=True),strict=False)
        print(f'loaded ckpt info: {info}')
def create_rwkv_args(transformer_config, config):
    from argparse import Namespace
    args = Namespace()
    args.layers = config['RWKV']['layers']
    args.my_pos_emb = 0
    args.head_size_a = 64
    args.head_size_divisor = 8
    args.ctx_len = 4096
    args.n_layer = transformer_config.num_hidden_layers
    args.n_embd = transformer_config.hidden_size
    args.dim_att = transformer_config.hidden_size
    args.dim_ffn = transformer_config.intermediate_size
    args.pre_ffn = 0
    args.head_qk = 0
    args.tiny_att_dim = 0
    args.tiny_att_layer = -999
    args.vocab_size = transformer_config.vocab_size
    args.pad_id = transformer_config.pad_token_id
    args.is_llama_ffn = config.get('is_llama_ffn',False)
    args.is_rwkv_att_only = config.get('is_rwkv_att_only',False)
    return args
         
if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer
    config_file = "configs/test_hybrid_full_logits_qwenmlp_local.yaml"
    import yaml
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    model_id = config['Llama']['model_id']
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    transformer_model = AutoModelForCausalLM.from_pretrained(model_id)
    print(transformer_model)
    args = create_rwkv_args(transformer_model.config, config)
    model = HybridModel(transformer_model,args)
    print(model)
    ckpt_file = '/home/yueyulin/model/qwen/qwen0.5Bepoch0/pytorch_model.bin'
    model.load_ckpt(ckpt_file)