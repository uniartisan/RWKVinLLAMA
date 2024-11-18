from torch.distributed import init_process_group
from torch.distributed import get_rank
import logging
import os
import time
from queue import Queue
from threading import Thread
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import threading
import traceback
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "DEBUG"),
    format='%(asctime)s.%(msecs)03d - PID:%(process)d : ThreadID:%(thread)d %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S %p'
)

logger = logging.getLogger(__name__)
from asyncio import Queue as AsyncQueue
import asyncio
class DistributedTeacher:
    def __init__(self, 
                 port, 
                 world_size, 
                 num_gpus, 
                 model_path, 
                 pad_id,
                 vocab_size,
                 batch_size,
                 max_length,
                 return_hiddens=True,
                 num_layers=28,
                 hidden_size=3584,
                 backend='nccl'):
        
        self.forward_counter_lock = threading.Lock()
        self.port = port
        self.world_size = world_size
        self.backend = backend
        self.num_gpus = num_gpus
        self.master_addr = '0.0.0.0'
        self.model_path = model_path
        self.model = None
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.io_threads = []
        self.model_threads = []
        self.running = True
        self.pad_id = pad_id
        self.return_hiddens = return_hiddens
        self._init_model()
        self._init_group()
        self.request_count = 0
        # 创建gather和scatter用的缓冲区
        self.dummy_input = torch.zeros((batch_size, max_length), dtype=torch.long, device='cuda:0')

        self.input_list = [
            torch.zeros(
                (batch_size, max_length),
                dtype=torch.long,
                device='cuda:0'
            )
            for _ in range(world_size)  # 包括rank 0的dummy tensor
        ]
        # logits的scatter缓冲区
        self.logits_scatter_list = [
            torch.zeros(
                (batch_size, max_length, vocab_size),
                dtype=torch.float32,
                device='cuda:0'
            )
            for _ in range(world_size)  # 包括rank 0的dummy tensor
        ]
        
        if return_hiddens:
            self.hidden_scatter_list = [
                torch.zeros(
                    ((num_layers+1)*batch_size, max_length, hidden_size),
                    dtype=torch.bfloat16,
                    device='cuda:0'
                )
                for _ in range(world_size)  # 包括rank 0的dummy tensor
            ]

    def _init_model(self):
        """Initialize the model across available GPUs"""
        device_map = {
            'model.embed_tokens': 2
        }
        layers_per_gpu = self.num_layers // (self.num_gpus-1)
        gpu_id = 1
        for i in range(self.num_layers):
            device_map[f'model.layers.{i}'] = gpu_id
            if (i+1) % layers_per_gpu == 0:
                gpu_id += 1
        device_map['lm_head'] = 2
        device_map['model.norm'] = 2
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                          device_map=device_map,
                                                          torch_dtype=torch.bfloat16,
                                                          attn_implementation="flash_attention_2")
        self.model.eval()
        logger.debug(f"Model initialized on {self.num_gpus} GPUs with dtype {self.model.dtype}")
        logger.debug(f'device_map: {self.model.hf_device_map}')
        with torch.no_grad():
            input_ids = torch.zeros((1, 1), dtype=torch.long, device='cuda:0')
            attention_mask = torch.ones((1, 1), dtype=torch.long, device='cuda:0')
            output = self.model(input_ids=input_ids, attention_mask=attention_mask,use_cache=False)
            logger.debug(f"Model output: {output}")

    def _init_group(self):
        """Initialize the distributed group"""
        logger.debug(f"Initializing process group with {self.world_size} processes")
        init_process_group(backend=self.backend,
                           init_method=f"tcp://localhost:{self.port}",
                           world_size=self.world_size,
                           rank=0)
        logger.debug(f"Process group initialized with {self.world_size} processes")
    
    def _forward_batch(self, input_ids_batch):
        """Forward一批请求"""
        torch.cuda.empty_cache()

        #loop process in serial
        for idx,input_ids in enumerate(input_ids_batch):
            attention_mask = torch.ne(input_ids, self.pad_id).to(input_ids.device)
            logger.debug(f"Forwarding batch of shape {input_ids.shape} for rank {idx+1}")
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False
                )
                self.logits_scatter_list[idx+1].copy_(outputs.logits)
                
                # hidden_states是一个list，每个元素形状为(batch_size, seq_len, hidden_size)
                if self.return_hiddens:
                    hidden_states = outputs.hidden_states
                    # 拼接所有层
                    hidden = torch.cat(hidden_states, dim=0)
                    # 形状为((num_layers+1)*batch_size, seq_len, hidden_size)
                    self.hidden_scatter_list[idx+1].copy_(hidden)
                    

        # concat所有请求 ATTENTION: RMSNorm is batch size dependent, WE CANNOT CONCAT in dorder to avoid batch size change
        # input_ids_batch = torch.cat(input_ids_batch, dim=0)
        # attention_mask = torch.ne(input_ids_batch, self.pad_id).to(input_ids_batch.device)
        # logger.debug(f"Forwarding batch of shape {input_ids_batch.shape}")
        # logger.debug(f"Value of input_ids_batch: {input_ids_batch}")
        # with torch.no_grad():
        #     outputs = self.model(
        #         input_ids=input_ids_batch,
        #         attention_mask=attention_mask,
        #         output_hidden_states=True,
        #         use_cache=False
        #     )
            

        #     logits = outputs.logits
            
        #     # 4. 复制每个rank的logits到scatter buffer
        #     for i in range(self.world_size-1):
        #         start_idx = i * self.batch_size
        #         end_idx = (i + 1) * self.batch_size
        #         self.logits_scatter_list[i+1].copy_(logits[start_idx:end_idx])
        #         logger.debug(f"scatter rank {i+1} logits to buffer, from {start_idx} to {end_idx}")
        #         logger.debug(f"""
        #         Rank {i+1} logits prepared:
        #         - Shape: {self.logits_scatter_list[i+1].shape}
        #         - Device: {self.logits_scatter_list[i+1].device},
        #         - logits value: {self.logits_scatter_list[i+1]}
        #         """)
        #         torch.cuda.synchronize()
        #     # 3. hidden_states是一个list，每个元素形状为(total_batch_size, seq_len, hidden_size)
        #     if self.return_hiddens:
        #         # 对每个rank处理
        #         for i in range(self.world_size-1):
        #             start_idx = i * self.batch_size
        #             end_idx = (i + 1) * self.batch_size
                    
        #             # 取出该rank对应的所有层的hidden states
        #             rank_hidden_states = [
        #                 h[start_idx:end_idx] for h in outputs.hidden_states
        #             ]  # 每个元素形状为(batch_size, seq_len, hidden_size)
                    
        #             # 拼接所有层
        #             rank_hidden = torch.cat(rank_hidden_states, dim=0)
        #             # 形状为((num_layers+1)*batch_size, seq_len, hidden_size)
                    
        #             self.hidden_scatter_list[i+1].copy_(rank_hidden)
        #             logger.debug(f"""
        #             Rank {i+1} hidden states prepared:
        #             - Batch range: {start_idx} to {end_idx}
        #             - Single layer shape: {rank_hidden_states[0].shape}
        #             - Concatenated shape: {rank_hidden.shape}
        #             """)
        #             torch.cuda.synchronize()

        logger.debug("Forwarded batch of shape for all ranks")

    
    
    def run_inference(self):
        """使用gather和scatter处理请求"""
        
        
        while self.running:
            self.request_count += 1
            try:
                # 1. Gather所有请求
                logger.debug("Gathering requests from all ranks")
                dist.gather(
                    tensor=self.dummy_input,
                    gather_list=self.input_list ,
                    dst=0
                )
                
                logger.debug(f"GPU memory before forward: {torch.cuda.memory_allocated()/1024**2}MB")
                # 3. 执行批量forward
                start_time = time.time()
                self._forward_batch(self.input_list[1:])
                logger.debug(f"Forward batch took {time.time() - start_time:.3f}s")
                
      
                # 5. Scatter logits
                logger.debug("Scattering logits to all ranks")
                dist.scatter(
                    tensor=self.logits_scatter_list[0],  # dummy tensor for rank 0
                    scatter_list=self.logits_scatter_list ,
                    src=0
                )
                
                # 6. 如果需要，准备hidden states的scatter_list
                if self.return_hiddens:
                    logger.debug("Scattering hidden states to all ranks")
                    dist.scatter(
                        tensor=self.hidden_scatter_list[0],
                        scatter_list=self.hidden_scatter_list,
                        src=0
                    )
                # 强制同步GPU
                torch.cuda.synchronize() 
                logger.debug(f"Completed one round of gather-forward-scatter for request {self.request_count}")
                
            except Exception as e:
                logger.error(f"Error in inference loop: {e}")
                traceback.print_exc()
                dist.barrier()
                continue

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--port', type=int, default=39500)
    argparser.add_argument('--world_size', type=int, default=5)
    argparser.add_argument('--num_gpus', type=int, default=3)
    argparser.add_argument('--model_path', type=str, default='/home/yueyulin/models/Qwen2.5-7B-Instruct/')
    argparser.add_argument('--pad_id', type=int, default=151645)
    argparser.add_argument('--batch_size', type=int, default=2)
    argparser.add_argument('--max_length', type=int, default=128)
    args = argparser.parse_args()
    config_file = os.path.join(args.model_path, 'config.json')
    with open(config_file) as f:
        import json
        configuration = json.load(open(config_file))
    print(configuration)
    vocab_size = configuration['vocab_size']
    hidden_size = configuration['hidden_size']
    num_layers = configuration['num_hidden_layers']
    batch_size = args.batch_size
    max_length = args.max_length
    # 设置NCCL环境变量
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["NCCL_NET_GDR_LEVEL"] = "5"
    dist_teacher = DistributedTeacher(args.port, 
                                      args.world_size, 
                                      args.num_gpus, 
                                      args.model_path, 
                                      args.pad_id,
                                      vocab_size,
                                      batch_size,
                                      max_length,
                                      return_hiddens=True,
                                      num_layers=num_layers,
                                      hidden_size=hidden_size,
                                      backend='nccl')
    dist_teacher.run_inference()