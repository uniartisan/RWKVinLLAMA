import ray
import argparse
import torch
import os
import signal
import sys
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_SOCKET_IFNAME"] = "eno1"  # 根据你的网络接口名称进行调整
os.environ["NCCL_IB_DISABLE"] = "1"        # 如果没有InfiniBand网络，可以禁用IB

#connect the args.host and args.port to the ray server
parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="192.168.1.39")
parser.add_argument("--port", type=int, default=6379)
parser.add_argument("--model_id", type=str, default="/home/yueyulin/models/Qwen2.5-32B-Instruct/")
parser.add_argument("--num_gpus", type=int, default=3)
parser.add_argument("--worker_name", type=str, default="worker")
args = parser.parse_args()
ret = ray.init(address=f"{args.host}:{args.port}")
print(ret)
from ray.util.queue import Queue
#Create a remote worker which will load the model with all nvidia gpus
@ray.remote(num_gpus=args.num_gpus)
class Worker:
    def __init__(self,model_id):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_id,device_map="auto",attn_implementation="flash_attention_2", low_cpu_mem_usage=True, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(self.model.hf_device_map)
        self.queue = Queue(maxsize=1)
        print(f"Worker {os.getpid()} is initialized")
    
    def forward(self,input_ids,return_hidden_states=False):
        attention_mask = torch.ne(input_ids,self.tokenizer.pad_token_id)
        self.queue.put(1)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 use_cache=False,
                                 output_hidden_states=return_hidden_states)
        self.queue.get()
        if return_hidden_states:
            return outputs.logits,outputs.hidden_states
        else:
            return outputs.logits
        

worker = Worker.options(
    name=args.worker_name
).remote(args.model_id)

print(worker)
# 定义信号处理函数
def signal_handler(sig, frame):
    print('接收到信号，正在退出...')
    ray.shutdown()
    sys.exit(0)

# 注册信号处理函数
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# 让主程序保持运行状态，等待信号
print('按Ctrl+C退出')
signal.pause()