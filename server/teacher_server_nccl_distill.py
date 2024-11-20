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

def do_model_forward(model,
                        input_ids,
                        eos_id,
                        return_hidden_states):
    logger.info(f"Start model forward with input_ids shape {input_ids.shape} input_ids {input_ids}")
    start_time = time.time()
    with torch.no_grad():
        attention_mask = torch.ne(input_ids,eos_id).to(input_ids.device)
        results = model(input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=return_hidden_states,
                        use_cache=False)
    logger.info(f"Finished model forward with logits shape {results.logits.shape} elapsed time is {time.time()-start_time}")
    torch.cuda.empty_cache()
    return results

def do_gather_scatter_thread(comm,
                           client_size,
                           batch_size,
                           length,
                           return_hidden_states,
                           model,
                           eos_id):
    global RUNNING
    logger.info(f"Start gather/scatter thread with {client_size} clients")
    
    with cp.cuda.Device(0):
        # Create gather buffer to receive inputs from all clients
        # Shape: (client_size, batch_size, length)
        gather_buffer = cp.empty((client_size, batch_size, length), dtype=cp.int64)
        stream = cp.cuda.Stream(non_blocking=True)
        #Logits buffer, shape (client_size, batch_size, length, vocab_size)
        logits_buffer = [torch.empty((batch_size, length, model.config.vocab_size), dtype=torch.float32, device='cuda:0') for _ in range(client_size)]
        #Hidden states buffer, shape (client_size, (num_layers+1), batch_size, length, hidden_size)
        if return_hidden_states:
            hidden_states_buffer = [torch.empty(((num_layers+1)* batch_size, length, model.config.hidden_size), dtype=torch.float32, device='cuda:0') for _ in range(client_size)]
        while RUNNING:
            # Gather input_ids from all clients at once
            logger.info("Starting gather operation from all clients")
            cp.cuda.nccl.groupStart()
            for i in range(client_size):
                comm.recv(gather_buffer[i].data.ptr, 
                         batch_size * length,
                         nccl.NCCL_INT64, 
                         i + 1,  # client ranks start from 1
                         stream.ptr)
            cp.cuda.nccl.groupEnd()
            logger.debug(f'waiting for {client_size} clients to send data')
            stream.synchronize()
            logger.info("Finished gathering input_ids from all clients")

            # Process all requests in batch
            results = []
            for i in range(client_size):
                input_ids = torch.as_tensor(gather_buffer[i], device='cuda:0', dtype=torch.long)
                result = do_model_forward(model, input_ids, eos_id, return_hidden_states)
                results.append(result)
            
            # Scatter results back to clients
            logger.info("Starting scatter operation to all clients")
            # First scatter logits
            cp.cuda.nccl.groupStart()
            for i in range(client_size):
                logger.debug(f"Scattering logits to client {i}")
                logger.debug(f'result[{i}].logits shape is {results[i].logits.shape} copy to logits_buffer[{i}], shape is {logits_buffer[i].shape}')
                logits = logits_buffer[i].copy_(results[i].logits)
                comm.send(logits.data_ptr(),
                         logits.size(0) * logits.size(1) * logits.size(2),
                         nccl.NCCL_FLOAT,
                         i + 1,
                         stream.ptr)
            cp.cuda.nccl.groupEnd()
            
            # If hidden states are requested, scatter them as well
            if return_hidden_states:
                cp.cuda.nccl.groupStart()
                for i in range(client_size):
                    hidden_states = results[i].hidden_states
                    for j,hidden_state in enumerate(hidden_states):
                            #copy the layer of hidden_state(batch_size,length,hidden_size) to the all_hidden_states_buffer[i]
                        hidden_states_buffer[i][j*batch:(j+1)*batch].copy_(hidden_state)
                    logger.debug(f"Scattering hidden states to client {i}, shape is {hidden_states_buffer[i].shape}")
                    comm.send(hidden_states_buffer[i].data_ptr(),
                            hidden_states_buffer[i].size(0) * hidden_states_buffer[i].size(1) * hidden_states_buffer[i].size(2),
                            nccl.NCCL_FLOAT,
                            i + 1,
                            stream.ptr)
                cp.cuda.nccl.groupEnd()
            del results
            stream.synchronize()
            logger.info("Finished scattering results to all clients")
            torch.cuda.empty_cache()  # Clean up GPU memory


def main(model_path,
         nccl_ids,
         num_gpus, 
         size, 
         batch,
         length,
         eos_id,
         num_layers=28,
         output_all_hiddens=False,
         device_map=None):
    print(f"Start server with model {model_path}, nccl_ids {nccl_ids}, num_gpus {num_gpus}, size {size}, batch {batch}, length {length}, eos_id {eos_id}, output_all_hiddens {output_all_hiddens}")
    """Initialize the model across available GPUs"""
    if device_map is None:
        device_map = {
                'model.embed_tokens': num_gpus-1
        }
        layers_per_gpu = num_layers // (num_gpus-1)
        gpu_id = 1
        for i in range(num_layers):
                device_map[f'model.layers.{i}'] = gpu_id
                if (i+1) % layers_per_gpu == 0: 
                    gpu_id += 1
        device_map['lm_head'] = num_gpus-1
        device_map['model.norm'] = num_gpus-1
    else:
        print(f"Loading device map from {device_map}")
        with open(device_map,'r') as f:
            device_map = json.load(f)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 device_map=device_map,
                                                 attn_implementation="flash_attention_2", 
                                                 low_cpu_mem_usage=True, 
                                                 torch_dtype=torch.bfloat16)
    print(f'finished loading model , placement of model is :{model.hf_device_map}')
    client_size = size - 1
    world_size = size 
    logger.info(f"Start creating nccl communicator with world_size {world_size}, nccl_ids {nccl_ids} on device 0")

    #Use device 0 to create the nccl communicator
    with cp.cuda.Device(0):
        COMM = nccl.NcclCommunicator(world_size,nccl_ids,0)

    logger.info(f"Finihsed creating nccl communicator with world_size {world_size}, nccl_ids {nccl_ids} on device 0")

    #The client rank starts from 1,2,3...world_size-1

    do_gather_scatter_thread(COMM,
                           client_size,
                           batch,
                           length,
                           output_all_hiddens,
                           model,
                           eos_id)

    # do_recv_send_thread(COMM,
    #                     client_size,
    #                     batch,
    #                     length,
    #                     output_all_hiddens,
    #                     model,
    #                     eos_id)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='model id')
    parser.add_argument('--batch', type=int, required=True, help='batch size')
    parser.add_argument('--length', type=int, required=True, help='length of input')
    parser.add_argument('--size', type=int, required=True, help='number of nodes')
    parser.add_argument('--output_all_hiddens', action='store_true',default=False, help='return all hiddens')
    parser.add_argument('--nccl_id_file', type=str, default='nccl.txt',help='nccl id file')
    parser.add_argument('--num_gpus', type=int, default=3, help='number of gpus to host the teacher')
    parser.add_argument('--num_layers', type=int, default=28, help='number of layers in the model')
    parser.add_argument('--device_map', type=str, default=None, help='device map')
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
    num_layers = args.num_layers
    main(args.model_id, nccl_ids,num_gpus,size,batch,length,eos_id,output_all_hiddens=args.output_all_hiddens,num_layers=num_layers,device_map=args.device_map)
