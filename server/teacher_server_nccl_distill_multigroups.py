import torch
import torch.amp
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import cupy as cp
from cupy.cuda import nccl
import time
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import threading
from queue import Queue
from torch.cuda.amp import autocast
import gc
import math

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format='%(asctime)s.%(msecs)03d | %(levelname)s | %(process)d | %(thread)d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('InferenceServer')
RUNNING = True

def log_gpu_memory():
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        logger.info(f"GPU {i} Memory: Allocated={alloc:.2f}GB, Reserved={reserved:.2f}GB")

class PipelineParallelModel:
    def __init__(self, model, num_gpus):
        self.model = model
        self.num_gpus = num_gpus
        self.pipeline_stages = self._create_pipeline_stages()
        self.forward_lock = threading.Lock()  # Add lock
        
    def _create_pipeline_stages(self):
        """Split model into pipeline stages - only called once during initialization"""
        logger.info("Initializing pipeline stages...")
        
        # Create ModuleList for each GPU
        pipeline_stages = [[] for _ in range(self.num_gpus)]
        
        # Move embedding to CPU
        self.model.model.embed_tokens.to('cuda:0')
        
        # Distribute layers across GPUs more evenly
        num_layers = len(self.model.model.layers)
        layers_per_gpu = math.ceil(num_layers // (self.num_gpus))
        
        # Distribute layers
        for i, layer in enumerate(self.model.model.layers):
            gpu_id = min(i // layers_per_gpu, self.num_gpus - 1)
            layer.to(f'cuda:{gpu_id}')
            pipeline_stages[gpu_id].append(layer)
        # Move final norm and head to last GPU
        self.model.model.norm.to(f'cuda:{self.num_gpus-1}')
        self.model.lm_head.to(f'cuda:{self.num_gpus-1}')
        
        logger.info("Pipeline stages initialized successfully")
        return pipeline_stages
    
    def forward(self, input_ids, attention_mask):
        """Execute forward pass collecting all hidden states"""
        # Move inputs to GPU:0
        input_ids = input_ids.to('cuda:0')
        attention_mask = attention_mask.to('cuda:0')
        
        # Initialize list to store hidden states
        all_hidden_states = []
        # attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Get embeddings on GPU:0
        hidden = self.model.model.embed_tokens(input_ids)
        
        # Process through pipeline stages
        with torch.amp.autocast(device_type='cuda',dtype=torch.bfloat16):
            for gpu_id, stage in enumerate(self.pipeline_stages):
                # Move hidden states and attention mask to current GPU
                old_hidden = hidden
                hidden = hidden.to(f'cuda:{gpu_id}')
                del old_hidden
                old_musk = attention_mask
                current_mask = attention_mask.to(f'cuda:{gpu_id}')
                del old_musk
                
                for layer in stage:
                    all_hidden_states.append(hidden.detach())
                    layer_outputs = layer(hidden, attention_mask=current_mask,use_cache=False)
                    hidden = layer_outputs[0]
                    del layer_outputs
                    # 删除不需要的变量
                del current_mask
                # 删除不需要的变量
            del attention_mask
            
            # Final norm and head on last GPU
            hidden = self.model.model.norm(hidden)
            all_hidden_states.append(hidden.detach())
            logits = self.model.lm_head(hidden).detach()

        del input_ids
        return logits, all_hidden_states

def init_recv_buffers(device_id, client_size, batch_size, length):
    with cp.cuda.Device(device_id):
        return [cp.empty((batch_size,length), dtype=cp.int64) for _ in range(client_size)]


def nccl_group_thread_runner(nccl_ids,
                             device_id,
                             world_size,
                             batch_size,
                             length,
                             vocab_size,
                             num_layers,
                             hidden_size,
                             eos_id,
                             return_hidden_states,
                             pipeline_model):
    torch.cuda.set_device(device_id)
    logger.info(f'Setting CUDA device to {device_id}')
    with cp.cuda.Device(device_id):
        # Initialize NCCL with server rank 0
        logger.info(f'Initializing NCCL communicator with nccl_ids {nccl_ids} and world_size {world_size} on device {device_id}')
        comm = nccl.NcclCommunicator(world_size, nccl_ids, 0)
        stream = cp.cuda.Stream(non_blocking=True)
        logger.info(f'NCCL communicator initialized with nccl_ids {nccl_ids} and world_size {world_size} on device {device_id}')
        # Initialize recv buffers
        recv_buffers = init_recv_buffers(device_id, world_size-1, batch_size, length)
        #logits buffer 
        logits_buffer = [torch.empty((batch_size,length,vocab_size),device=f'cuda:{device_id}',dtype=torch.float32) for _ in range(world_size-1)]
        logger.info(f'Initialized recv buffers with size {len(recv_buffers)}')
        if return_hidden_states:
            #init hidden states buffer
            all_hidden_states_buffer = []
            for i in range(world_size-1):
                hidden_states_buffer = [torch.empty((batch_size,length,hidden_size),device=f'cuda:{device_id}',dtype=torch.float32) for _ in range(num_layers+1)]
                hidden_states_buffer = torch.cat(hidden_states_buffer,dim=0)
                all_hidden_states_buffer.append(hidden_states_buffer)
                logger.info(f'Initialized hidden states buffer with size {hidden_states_buffer.shape}')
            logger.info(f'Initialized hidden states buffer with size {len(all_hidden_states_buffer)}')
        # Initialize sleep parameters
        sleep_time = 0.001  # 1ms sleep between polling

   
        while True:
            try:
                torch.cuda.empty_cache()
                # start to receive from clients
                for i in range(world_size-1):
                    logger.info(f'Receiving from client {i} with {nccl_ids}')
                    comm.recv(recv_buffers[i].data.ptr, batch_size*length, nccl.NCCL_INT64, i+1, stream.ptr)
                logger.info(f'Synchronize stream to wait for all receives to finish')
                stream.synchronize()
                logger.info(f'Finished receiving from all clients')
                # do the forward pass one by one to make the logits and hidden states same as local
                for i in range(world_size-1):
                    input_ids = torch.as_tensor(recv_buffers[i],dtype=torch.long)
                    attention_mask = torch.ne(input_ids, eos_id)
                    logger.debug(f'input_ids : {input_ids}')
                    with pipeline_model.forward_lock:
                        logits,hidden_states = pipeline_model.forward(input_ids,attention_mask)
                    logger.debug(f'finished forward pass for client {i}')
                    del input_ids,attention_mask
                    logits_buffer[i].copy_(logits)
                    del logits
                    if return_hidden_states:
                        for j,hidden_state in enumerate(hidden_states):
                            #copy the layer of hidden_state(batch_size,length,hidden_size) to the all_hidden_states_buffer[i]
                            all_hidden_states_buffer[i][j*batch:(j+1)*batch].copy_(hidden_state)
                        del hidden_states
                # send the logits back to clients
                for i in range(world_size-1):
                    comm.send(logits_buffer[i].data_ptr(), batch_size*length*vocab_size, nccl.NCCL_FLOAT32, i+1, stream.ptr)
                    logger.info(f'Finished sending logits back to client {i}')
                    if return_hidden_states:
                        comm.send(all_hidden_states_buffer[i].data_ptr(), all_hidden_states_buffer[i].shape[0]*all_hidden_states_buffer[i].shape[1]*all_hidden_states_buffer[i].shape[2], nccl.NCCL_FLOAT32, i+1, stream.ptr)
                        logger.info(f'Finished sending hidden states back to client {i} with size :{all_hidden_states_buffer[i].shape}')
                stream.synchronize()
                torch.cuda.synchronize(device_id)
                gc.collect()
                logger.info(f'Finished sending all logits back to clients')
            except Exception as e:
                    logger.error(f'Error in device {device_id} processing loop: {e}', exc_info=True)
                    # Continue the while loop to keep serving requests
                    continue
def main(model_path,
         arr_nccl_ids,
         num_gpus, 
         size, 
         batch,
         length,
         eos_id,
         output_all_hiddens=False):
    logger.info(f'Initializing pipeline parallel model with model_path {model_path}')
    
    # Load model without device_map first
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                torch_dtype=torch.bfloat16,
                                                attn_implementation='flash_attention_2',)
    model.eval()
    hidden_size = model.config.hidden_size
    num_layers = len(model.model.layers)
    vocab_size = model.config.vocab_size
    # Initialize pipeline parallel model once
    pipeline_model = PipelineParallelModel(model, num_gpus)
    print('Finished initializing pipeline parallel model')
    threads = []
    for thread_id in range(len(arr_nccl_ids)):
        nccl_ids = arr_nccl_ids[thread_id]
        thread = threading.Thread(target=nccl_group_thread_runner, args=(nccl_ids,
                                                                         thread_id,
                                                                         size,
                                                                         batch,
                                                                         length,
                                                                         vocab_size,
                                                                         num_layers,
                                                                         hidden_size,
                                                                         eos_id,
                                                                         output_all_hiddens,
                                                                         pipeline_model))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='model id')
    parser.add_argument('--batch', type=int, required=True, help='batch size')
    parser.add_argument('--length', type=int, required=True, help='length of input')
    parser.add_argument('--size', type=int, required=True, help='number of nodes of each group')
    parser.add_argument('--output_all_hiddens', action='store_true', default=False, help='return all hiddens')
    parser.add_argument('--nccl_id_file', type=str, default='nccl.txt', help='nccl id file')
    parser.add_argument('--num_gpus', type=int, default=3, help='number of gpus to host the teacher')
    parser.add_argument('--chunk_size', type=int, default=2, help='chunk size for pipeline parallel processing')
    parser.add_argument('--num_nccl_groups', type=int, default=2, help='number of nccl groups')
    args = parser.parse_args()
    
    # Create and save nccl_ids
    arr_nccl_ids = []
    for i in range(args.num_nccl_groups):
        nccl_id_file = f'nccl.{i}.txt'
        with open(nccl_id_file, 'w') as f:
            nccl_ids = nccl.get_unique_id()
            arr_nccl_ids.append(nccl_ids)
            import json
            json.dump({'nccl_ids': nccl_ids}, f)
    
    batch = args.batch
    length = args.length
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    eos_id = tokenizer.pad_token_id
    num_gpus = args.num_gpus
    size = args.size
    main(args.model_id, 
         arr_nccl_ids,
         num_gpus,
         size,
         batch,
         length,
         eos_id,
         output_all_hiddens=args.output_all_hiddens)