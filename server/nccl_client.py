import torch
import cupy as cp
from cupy.cuda import nccl
import os
import logging
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('InferenceClient')
class InferenceClient:
    def __init__(self,
                 world_size, 
                 global_rank,
                 local_rank,
                 nccl_id,
                 batch_size,
                 length,
                 vocab_size,
                 num_layers,
                 hidden_size,
                 output_hidden_states = True):
        """
        初始化NCCL客户端
        Args:
            world_size: 所有GPU数量
            rank: 当前Client的rank，所偶的rank都必须大于等于num_server_rank
            num_server_rank: 服务器节点的rank,
            nccl_id: nccl id,
            batch_size: batch size,
            length: 输入的长度
            vocab_size: vocab size
            num_layers: number of layers
            hidden_size: hidden size
            output_hidden_states: 是否输出hidden states

            初始化nccl communicator
        """
        self.world_size = world_size
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.num_server_rank = 0
        self.nccl_id = nccl_id
        self.last_request_server_rank = None
        self.batch_size = batch_size
        self.length = length
        self.output_hidden_states = output_hidden_states
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.request_count = 0
        self._connect()
        logger.info(f"Finished initializing client {self.local_rank}, world size: {self.world_size}, global rank: {self.global_rank}, local rank: {self.local_rank}, num server rank: {self.num_server_rank}, nccl id: {self.nccl_id}, batch size: {self.batch_size}, length: {self.length}, vocab size: {self.vocab_size}, num layers: {self.num_layers}, hidden size: {self.hidden_size}, output hidden states: {self.output_hidden_states}")
    def _connect(self):
        """初始化NCCL communicator"""
        cp.cuda.Device(self.local_rank).use()
        print(f"Rank {self.local_rank} initializing communicator/nccl_id: {self.nccl_id}/{self.global_rank}/{self.world_size}")
        self.comm = nccl.NcclCommunicator(self.world_size, self.nccl_id, self.global_rank)
        self.stream = cp.cuda.Stream(non_blocking=True)
        logger.info(f"Rank {self.local_rank} initialized communicator")
        #init the recv buffer
        #logits shape is (batch_size,length,vocab_size)
        self.logits_buffer = cp.empty((self.batch_size, self.length,self.vocab_size), dtype=cp.float32)
        if self.output_hidden_states:
            #hidden_states shape is [num_layers,batch_size,length,hidden_size]
            self.hidden_states_buffer = cp.empty(((self.num_layers+1)*self.batch_size, self.length,self.hidden_size), dtype=cp.float32)


    def forward(self, input_ids,output_hidden_states = True):
        """
        Args:
            input_ids: shape is (batch_size,length)
        """
        #select the server rank according to the request count
        server_rank = 0
        with cp.cuda.Device(self.local_rank):
            input_ids = cp.asarray(input_ids, dtype=cp.int64)
            #send the input_ids to the server
            logger.info(f"Rank {self.local_rank} sending input_ids to server {server_rank} with shape {input_ids.shape}")
            self.comm.send(input_ids.data.ptr, input_ids.shape[0]*input_ids.shape[1], nccl.NCCL_INT64, server_rank, self.stream.ptr)
            self.stream.synchronize()
            #receive the logits from the server
            logger.info(f"Rank {self.local_rank} receiving logits from server {server_rank}")
            self.comm.recv(self.logits_buffer.data.ptr, self.logits_buffer.size, nccl.NCCL_FLOAT, server_rank, self.stream.ptr)
            self.stream.synchronize()
            logits = torch.as_tensor(self.logits_buffer, device=f'cuda:{self.local_rank}', dtype=torch.float32)
            logger.info(f"Rank {self.local_rank} received logits from server {server_rank}, shape is {logits.shape}")
            if self.output_hidden_states:
                #receive the hidden_states from the server
                logger.info(f"Rank {self.local_rank} receiving hidden states from server {server_rank},size is {self.hidden_states_buffer.size}with {self.num_layers+1} * {self.batch_size} * {self.length} * {self.hidden_size}")
                self.comm.recv(self.hidden_states_buffer.data.ptr, self.hidden_states_buffer.size, nccl.NCCL_FLOAT, server_rank, self.stream.ptr)
                self.stream.synchronize()
                logger.info(f"Rank {self.local_rank} received hidden states from server {server_rank}")
                hidden_states = torch.as_tensor(self.hidden_states_buffer, device=f'cuda:{self.local_rank}', dtype=torch.float32)
                logger.info(f"Rank {self.local_rank} hidden states shape: {hidden_states.shape}")
                return logits, hidden_states
            return logits

if __name__ == "__main__":
    #find the model configuration from the model path
    model_path = '/home/yueyulin/models/Qwen2.5-14B-Instruct/'
    config_path = os.path.join(model_path,'config.json')
    import json
    with open(config_path,'r') as f:
        config = json.load(f)
    vocab_size = config['vocab_size']
    num_layers = config['num_hidden_layers']
    hidden_size = config['hidden_size']


    batch = 1
    length = 256
    output_hidden_states = True
    nccl_file = 'nccl.txt'
    with open(nccl_file,'r') as f:
        import json 
        nccl_id = json.load(f)['nccl_ids']
        nccl_id = tuple(nccl_id)
    logger.info(f"NCCL ID: {nccl_id}")

    world_size = 2
    rank = 1
    num_server_rank = 1
    local_rank = 0

    logger.info(f'batch size: {batch}, length: {length}, vocab size: {vocab_size}, num layers: {num_layers}, hidden size: {hidden_size}')
    logger.info(f'world size: {world_size}, rank: {rank}, num server rank: {num_server_rank}, local rank: {local_rank}')

    client = InferenceClient(
        world_size = world_size,
        global_rank = rank,
        local_rank = local_rank,
        nccl_id = nccl_id,
        batch_size = batch,
        length = length,
        vocab_size = vocab_size,
        num_layers = num_layers,
        hidden_size = hidden_size,
        output_hidden_states = output_hidden_states
    )

    import time
    while True:
        input('Press to start sending requests')
        input_ids = torch.randint(0, vocab_size-1, (batch, length), dtype=torch.long).to(rank)
        logger.info(f"Input ids shape: {input_ids.shape}")
        start_time = time.time()
        logits, hidden_states = client.forward(input_ids)
        logger.info(f"Time elapsed: {time.time()-start_time}")
        logger.info(f"Logits shape: {logits.shape}")
        logger.info(f"Hidden states shape: {hidden_states.shape}")