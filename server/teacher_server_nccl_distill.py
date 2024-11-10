import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
import argparse
import cupy as cp
from cupy.cuda import nccl
import time
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import threading
from queue import Queue
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('InferenceServer')
RUNNING = True
nccl_lock_for_comms = []
def handle_request_sync(model, input_ids,eos_id,output_all_hiddens=False):
    logger.info(f"Start inference,input_ids shape is {input_ids.shape}, eos_id is {eos_id} input_ids {input_ids},output_all_hiddens is {output_all_hiddens}")  
    start_time = time.time()
    with torch.no_grad():
        attention_mask = torch.ne(input_ids,eos_id).to(input_ids.device)
        results =  model(input_ids,
                         attention_mask=attention_mask,
                         output_hidden_states=output_all_hiddens,
                         use_cache=False)
    logger.info(f"Finished inference,result logits shape is {results.logits.shape} elapsed time is {time.time()-start_time}")
    return results.logits,results.hidden_states

def do_model_forward(model,
                        input_ids,
                        eos_id,
                        return_hidden_states):
    logger.info(f"Start model forward with input_ids shape {input_ids.shape}")
    start_time = time.time()
    with torch.no_grad():
        attention_mask = torch.ne(input_ids,eos_id).to(input_ids.device)
        results = model(input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=return_hidden_states,
                        use_cache=False)
    logger.info(f"Finished model forward with logits shape {results.logits.shape} elapsed time is {time.time()-start_time}")
    return results

def do_recv_send_thread(comm,
                        client_size,
                        batch_size,
                        length,
                        return_hidden_states,
                        model,
                        eos_id
                        ):
    global RUNNING
    logger.info(f"Start receiving requests from {client_size} clients in serial mode")
    with cp.cuda.Device(0):
        recv_buffers = [cp.empty((batch_size,length),dtype=cp.int64) for _ in range(client_size)]
        stream = cp.cuda.Stream(non_blocking=True)
        while RUNNING:
            #recv buffers from all clients
            for i in range(client_size):
                comm.recv(recv_buffers[i].data.ptr,batch_size*length,nccl.NCCL_INT64,i+1,stream.ptr)
            stream.synchronize()
            logger.info(f"Finished receiving requests from clients{client_size}")
            results = []
            for i in range(client_size):
                input_ids = torch.as_tensor(recv_buffers[i],device=f'cuda:0',dtype=torch.long)
                result = do_model_forward(model,input_ids,eos_id,return_hidden_states)
                results.append(result)
            #send the results back to the clients
            for i in range(client_size):
                logits = results[i].logits.to(f'cuda:0').contiguous()
                logger.info(f"Start send logits to client {i+1} with logits shape {logits.shape} device is {logits.device}")
                comm.send(logits.data_ptr(),logits.size(0)*logits.size(1)*logits.size(2),nccl.NCCL_FLOAT,i+1,stream.ptr)
                logger.info(f"Finished send logits to client {i+1}")
                if return_hidden_states:
                    hidden_states = results[i].hidden_states
                    hidden_states = torch.cat(hidden_states,dim=0).to(device=f'cuda:0',dtype=torch.float32).contiguous()
                    logger.info(f"Start send hidden_states to client {i+1} with hidden_states shape {hidden_states.shape} hidden_states device is {hidden_states.device} sending size is {hidden_states.size(0)*hidden_states.size(1)*hidden_states.size(2)}")
                    comm.send(hidden_states.data_ptr(),hidden_states.size(0)*hidden_states.size(1)*hidden_states.size(2),nccl.NCCL_FLOAT,i+1,stream.ptr)
                    logger.info(f"Finished send hidden_states to client {i+1}")
            stream.synchronize()                    
            


#Thread function to receive the requests from specific rank of client
def request_reciver_thread(comm,
                           server_rank,
                           client_rank,
                           batch_size,
                           length,
                           request_queue,
                           return_hidden_states):
    global RUNNING
    logger.info(f"Start receiving requests from rank {client_rank} of comm {server_rank}")
    nccl_lock = nccl_lock_for_comms[server_rank]
    with cp.cuda.Device(server_rank):
        #recv_buffer is used to store (batch_size,length) int64 tensor
        recv_buffer = cp.empty((batch_size,length),dtype=cp.int64)
        stream = cp.cuda.Stream(non_blocking=True)
        event = cp.cuda.Event()  # 创建一个 CUDA 事件
        while RUNNING:
            #receive the request from the client
            logger.info(f"Start receive request from rank {client_rank} of comm {server_rank} with data shape {batch_size}*{length}")
            with nccl_lock:
                comm.recv(recv_buffer.data.ptr,batch_size*length,nccl.NCCL_INT64,client_rank,stream.ptr)

            logger.info(f"Received request from rank {client_rank} of comm {server_rank}")
            stream.synchronize()
            logger.info(f"Stream synchronized")
            #put the request into the server_rank
            input_ids = torch.as_tensor(recv_buffer,device=f'cuda:{server_rank}',dtype=torch.long)
            condition = threading.Condition(nccl_lock)
            result = []
            request_queue.put({
                'input_ids':input_ids,
                'condition':condition,
                'result':result
            })
            logger.info(f"Put request from rank {client_rank} into request_queue with data: {input_ids.shape}/from {client_rank},comm {comm}")

            #wait for the request to be consumed
            with condition:
                condition.wait()
            logger.info(f"Request from rank {client_rank} is consumed")
            results = result[0]
            logits = results.logits.to(f'cuda:{server_rank}').contiguous()
            #send the response back to the client
            with nccl_lock:
                logger.info(f"Start send response to rank {client_rank} of comm {server_rank} with logits shape {logits.shape}")
                comm.send(logits.data_ptr(),logits.size(0)*logits.size(1)*logits.size(2),nccl.NCCL_FLOAT,client_rank,stream.ptr)
            if return_hidden_states:
                hidden_states = results.hidden_states#(num_layers,batch_size,length,hidden_size)
                hidden_states = torch.cat(hidden_states,dim=0).to(f'cuda:{server_rank}').contiguous()
                logger.info(f"Start send hidden_states to rank {client_rank} of comm {server_rank} with hidden_states shape {hidden_states.shape}")
                with nccl_lock:
                    comm.send(hidden_states.data_ptr(),hidden_states.size(0)*hidden_states.size(1)*hidden_states.size(2),nccl.NCCL_FLOAT,client_rank,stream.ptr)
            stream.synchronize()
            logger.info(f"Finished send response to rank {client_rank} of comm {server_rank}")
#Thread function to do the model forward
def model_forward_thread(model,
                            request_queue,
                            eos_id,
                            return_hidden_states):
    global RUNNING
    logger.info(f"Start model forward thread")
    while RUNNING:
        request = request_queue.get()
        input_ids = request['input_ids']
        condition = request['condition']
        result = request['result']
        attention_mask = torch.ne(input_ids,eos_id).to(input_ids.device)
        logger.info(f"Start model forward with input_ids shape {input_ids.shape}")
        with torch.no_grad():
            results = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=return_hidden_states
            )
        logger.info(f"Finished model forward with logits shape {results.logits.shape}")
        result.append(results)
        with condition:
            condition.notify()


def main(model_path,
         nccl_ids,
         num_gpus, 
         size, 
         batch,
         length,
         eos_id,
         output_all_hiddens=False):
    print(f"Start server with model {model_path}, nccl_ids {nccl_ids}, num_gpus {num_gpus}, size {size}, batch {batch}, length {length}, eos_id {eos_id}, output_all_hiddens {output_all_hiddens}")
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 device_map="auto",
                                                 attn_implementation="flash_attention_2", 
                                                 low_cpu_mem_usage=True, 
                                                 torch_dtype=torch.float16)
    print(f'finished loading model , placement of model is :{model.hf_device_map}')
    client_size = size - 1
    world_size = size 

    #Use device 0 to create the nccl communicator
    with cp.cuda.Device(0):
        COMM = nccl.NcclCommunicator(world_size,nccl_ids,0)

    logger.info(f"Finihsed creating nccl communicator with world_size {world_size}, nccl_ids {nccl_ids} on device 0")

    #The client rank starts from 1,2,3...world_size-1
    do_recv_send_thread(COMM,
                        client_size,
                        batch,
                        length,
                        output_all_hiddens,
                        model,
                        eos_id)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='model id')
    parser.add_argument('--batch', type=int, required=True, help='batch size')
    parser.add_argument('--length', type=int, required=True, help='length of input')
    parser.add_argument('--size', type=int, required=True, help='number of nodes')
    parser.add_argument('--output_all_hiddens', action='store_true',default=False, help='return all hiddens')
    parser.add_argument('--nccl_id_file', type=str, default='nccl.txt',help='nccl id file')
    parser.add_argument('--num_gpus', type=int, default=3, help='number of gpus to host the teacher')
    args = parser.parse_args()
    
    nccl_id_file = args.nccl_id_file
    #create nccl_ids and save them to the file 
    with open(nccl_id_file,'w') as f:
        nccl_ids = nccl.get_unique_id()
        import json
        json.dump({'nccl_ids':nccl_ids},f)
    
    batch = args.batch
    length = args.length
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    eos_id = tokenizer.pad_token_id
    num_gpus = args.num_gpus
    size = args.size
    main(args.model_id, nccl_ids,num_gpus,size,batch,length,eos_id,output_all_hiddens=args.output_all_hiddens)