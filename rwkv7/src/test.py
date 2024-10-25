import sys
import os
import torch
from model import RWKV_Tmix_x070

print(RWKV_Tmix_x070)
from argparse import Namespace
args = Namespace(n_embd=512, n_layer=6, dim_att=512,head_size_a=64,head_size_divisor=8)
rwkv7 = RWKV_Tmix_x070(args, 0).bfloat16().cuda()
print(rwkv7)

B,T,C = 10,256,512
x = torch.randn(B,T,C).bfloat16().cuda()
print(x)
y = rwkv7(x)
print(y.shape)
print(y)