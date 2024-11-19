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
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "DEBUG"),
    format='%(asctime)s.%(msecs)03d - PID:%(process)d : ThreadID:%(thread)d %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S %p'
)

logger = logging.getLogger(__name__)

class StudentClient:
    def __init__(self,
                master_addr,
                master_port,
                world_size,
                device_id,
                rank,
                return_hiddens,
                batch_size,
                num_layers,
                hidden_size,
                max_length,
                vocab_size,
                backend='nccl'):
        self.master_addr = master_addr
        self.master_port = master_port
        self.world_size = world_size
        self.device_id = device_id
        self.rank = rank
        self.backend = backend
        self.return_hiddens = return_hiddens
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.output_logits = torch.zeros(
            (self.batch_size, self.max_length, self.vocab_size),
            dtype=torch.float32,
            device=f'cuda:{self.device_id}'
        )
        
        if self.return_hiddens:
            self.output_hidden = torch.zeros(
                ((self.num_layers+1)*self.batch_size, self.max_length, self.hidden_size),
                dtype=torch.bfloat16,
                device=f'cuda:{self.device_id}'
            )
    def init_process_group(self):
        os.environ["NCCL_IB_DISABLE"] = "0"
        os.environ["NCCL_NET_GDR_LEVEL"] = "5"
        logger.info(f'Initializing process group with backend: {self.backend} with rank {self.rank} and world size {self.world_size} on device {self.device_id}')
        init_process_group(
            backend=self.backend,
            init_method=f'tcp://{self.master_addr}:{self.master_port}',
            rank=self.rank,
            world_size=self.world_size
        )
        logger.info(f'Process group initialized with rank {self.rank} and world size {self.world_size}')

    
    
    def forward(self, input_ids):
        """使用gather/scatter但只保存自己的结果"""
        try:
            # 参与gather - 注意不需要gather_list
            dist.gather(
                tensor=input_ids,  # 发送自己的输入
                gather_list=None,  # client不需要gather_list
                dst=0              # 发送给rank 0
            )

            dist.scatter(
                tensor=self.output_logits,  # 使用预分配的tensor
                scatter_list=None,     # client不需要scatter_list
                src=0                  # 从rank 0接收
            )


            if self.return_hiddens:
                dist.scatter(
                    tensor=self.output_hidden,  # 使用预分配的hidden states tensor
                    scatter_list=None,
                    src=0
                )
                
                torch.cuda.synchronize()
                return self.output_logits, self.output_hidden   
            torch.cuda.synchronize()    
            return self.output_logits

        except Exception as e:
            logger.error(f"Error in client forward: {e}")
            raise e
def run_student(rank, world_size, master_addr, master_port, model_path):
    """Initialize and run a student process"""
    configuration_file = os.path.join(model_path, 'config.json')
    with open(configuration_file, 'r') as f:
        import json
        config = json.load(f)
    vocab_size = config['vocab_size']
    hidden_size = config['hidden_size']
    num_hidden_layers = config['num_hidden_layers']
    # Set device for this process
    device_id = rank - 1  # Rank 0 is teacher, so subtract 1 for device mapping
    torch.cuda.set_device(device_id)
    batch_size = 2
    max_length = 1024

    # Create student client
    student = StudentClient(
        master_addr=master_addr,
        master_port=master_port,
        world_size=world_size,
        device_id=device_id,
        rank=rank,
        return_hiddens=True,
        batch_size=batch_size,
        num_layers=num_hidden_layers,
        hidden_size=hidden_size,
        max_length=max_length,
        vocab_size=vocab_size,
        backend='nccl'
    )

    
    # Initialize process group
    student.init_process_group()
    
    for i in range(100000):
        # Generate random input ids
        input_ids = torch.randint(
            0, vocab_size, 
            (batch_size, max_length), 
            device=f'cuda:{device_id}'
        )
        
        # Forward pass
        logger.info(f"Rank {rank}: Starting forward pass")
        # 根据return_hiddens接收不同的返回值
        start_time = time.time()
        if student.return_hiddens:
            logits, hidden_states = student.forward(input_ids=input_ids)
        else:
            logits = student.forward(input_ids=input_ids)
        end_time = time.time()
        
        logger.info(f"Rank {rank}: Completed forward pass, time taken: {end_time - start_time:.4f} seconds")
        logger.info(f"Rank {rank}: Logits shape: {logits.shape}")
        if student.return_hiddens:
            logger.info(f"Rank {rank}: Hidden states shape: {hidden_states.shape}")

    print(f"Rank {rank} finished")

def main():
    os.environ['NCCL_DEBUG'] = 'WARN'
    parser = argparse.ArgumentParser()
    parser.add_argument('--master_addr', type=str, default='192.168.1.39')
    parser.add_argument('--master_port', type=int, default=39500)
    parser.add_argument('--world_size', type=int, default=2)  # 1 teacher + 4 students
    parser.add_argument('--num_gpus', type=int, default=4)
    parser.add_argument('--model_path', type=str, default='/home/yueyulin/models/Qwen2.5-7B-Instruct/')
    parser.add_argument('--pad_id', type=int, default=151645)
    args = parser.parse_args()
    import multiprocessing as mp


    # Start student processes
    processes = []
    for rank in range(1, args.world_size):  # ranks 1-4 are students
        p = mp.Process(
            target=run_student,
            args=(
                rank,
                args.world_size,
                args.master_addr,
                args.master_port,
                args.model_path
            )
        )
        p.start()
        processes.append(p)

    while True:
        quit = input("Press Enter to continue...")
        if quit == 'q':
            break

    # Wait for all processes to complete
    for p in processes:
        p.terminate()
if __name__ == "__main__":
    
    main()