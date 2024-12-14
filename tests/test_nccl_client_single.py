import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(process)d | %(thread)d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('InferenceTest')

def validate_group_results(model, input_ids, attention_mask, results, start_rank, batch):
    """Validate results for one NCCL group"""
    try:
        model.eval()
        for i, result in enumerate(results):
            if result is None:
                logger.warning(f"Skipping validation for device {i} - no results")
                continue
                
            thread_logits, thread_hidden_states = result
            batch_start = i * batch
            batch_end = (i + 1) * batch
            
            # Move model to correct device for this batch
            device = f'cuda:{start_rank + i}'
            model.to(device)
            
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    local_output = model(
                        input_ids[batch_start:batch_end].to(device),
                        attention_mask=attention_mask[batch_start:batch_end].to(device),
                        output_hidden_states=True
                    )
            
            local_logits = local_output.logits.float().cpu()
            thread_logits = thread_logits.float()
            logits_diff = torch.max(torch.abs(thread_logits - local_logits)).item()
            
            local_hidden = torch.cat([h.float().cpu() for h in local_output.hidden_states], dim=0)
            thread_hidden = thread_hidden_states.float()
            hidden_diff = torch.max(torch.abs(thread_hidden - local_hidden)).item()
            
            logger.info(f"Group device {start_rank + i} validation:")
            logger.info(f"  Max logits difference: {logits_diff}")
            logger.info(f"  Max hidden states difference: {hidden_diff}")
            
            if logits_diff >= 1e-3 or hidden_diff >= 1e-3:
                logger.error(f"Validation failed for device {start_rank + i}")
                return False
                
        return True
            
    except Exception as e:
        logger.error(f"Error during validation: {e}", exc_info=True)
        return False

def run_group_inference(group_id, nccl_file, model_path, input_texts, vocab_size, num_layers, 
                       hidden_size, world_size, batch, length, output_hidden_states, start_rank):
    """Run inference for one NCCL group"""
    
    logger.info(f"Starting group {group_id} inference with devices {start_rank}-{start_rank+1}")
    
    try:
        # Load NCCL ID
        with open(nccl_file, 'r') as f:
            nccl_id = tuple(json.load(f)['nccl_ids'])
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        input_ids = tokenizer(input_texts, 
                            return_tensors='pt',
                            padding="max_length",
                            max_length=length,
                            truncation=True).input_ids
        attention_mask = torch.ne(input_ids, tokenizer.pad_token_id)
        
        # Initialize model for validation
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2'
        )
        model.eval()

        results = [None] * (world_size - 1)  # For 2 client devices per group
        clients = [None] * (world_size - 1)

        def client_worker(index):
            """Worker function for each client device"""
            device_id = start_rank + index
            torch.cuda.set_device(device_id)
            
            try:
                logger.info(f"Group {group_id} - Starting client on device {device_id}")
                if clients[index] is None:
                
                    from server.nccl_client import InferenceClient
                    client = InferenceClient(
                        nccl_id=nccl_id,
                        world_size=world_size,
                        batch_size=batch,
                        length=length,
                        output_hidden_states=output_hidden_states,
                        local_rank=device_id,
                        global_rank=index + 1,
                        vocab_size=vocab_size,
                        num_layers=num_layers,
                        hidden_size=hidden_size
                    )
                    clients[index] = client    
                my_input_ids = input_ids[index * batch:(index + 1) * batch]
                torch.cuda.synchronize(device_id)
                logits, hidden_states = client.forward(my_input_ids)
                torch.cuda.synchronize(device_id)
                
                results[index] = (logits.cpu(), hidden_states.cpu())
                logger.info(f"Group {group_id} - Device {device_id} completed successfully")
                
            except Exception as e:
                logger.error(f"Group {group_id} - Error on device {device_id}: {e}", exc_info=True)
                results[index] = None
            finally:
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device_id)

        # Start client threads for this group
        threads = []
        for i in range(world_size - 1):
            thread = threading.Thread(target=client_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads in this group
        for thread in threads:
            thread.join()
            
        # Validate results
        if any(result is not None for result in results):
            if validate_group_results(model, input_ids, attention_mask, results, start_rank, batch):
                logger.info(f"Group {group_id} validation successful")
            else:
                logger.error(f"Group {group_id} validation failed")
        
        return results, input_ids

    except Exception as e:
        logger.error(f"Error in group {group_id}: {e}", exc_info=True)
        return None

def main():
    # Configuration
    model_path = '/home/yueyulin/models/Qwen2.5-7B-Instruct/'
    
    # Different input texts for each group
    group_texts = {
        0: [
            'Long long ago, there was a beautiful princess.',
            '你好，我们能成为朋友吗？如果可以，请告诉我你的名字。',
            'The quick brown fox jumps over the lazy dog.',
            'I am a student at MIT.'
        ],
        1: [
            '请问怎么才能回到过去？',
            '从前有座山，山上有座庙，庙里有个老和尚和小和尚。',
            'Once upon a time, there was a beautiful princess.',
            'I am a student at Harvard University.'
        ]
    }
    
    # Load model config
    with open(os.path.join(model_path, 'config.json'), 'r') as f:
        config = json.load(f)
    
    # Common parameters
    params = {
        'model_path': model_path,
        'vocab_size': config['vocab_size'],
        'num_layers': config['num_hidden_layers'],
        'hidden_size': config['hidden_size'],
        'world_size': 3,  # 1 server + 2 clients per group
        'batch': 2,
        'length': 128,
        'output_hidden_states': True
    }
    
    # Group configurations
    groups = [
        {
            'group_id': 0,
            'nccl_file': 'nccl.0.txt',
            'start_rank': 0,
            'input_texts': group_texts[0]
        },
        {
            'group_id': 1,
            'nccl_file': 'nccl.1.txt',
            'start_rank': 2,
            'input_texts': group_texts[1]
        }
    ]
    
    # Run inference for each group in parallel
    def run_group(group):
        return run_group_inference(
            group['group_id'],
            group['nccl_file'],
            params['model_path'],
            group['input_texts'],
            params['vocab_size'],
            params['num_layers'],
            params['hidden_size'],
            params['world_size'],
            params['batch'],
            params['length'],
            params['output_hidden_states'],
            group['start_rank']
        )
    
    threads = []
    group_results = [None] * len(groups)
    
    for i, group in enumerate(groups):
        thread = threading.Thread(
            target=lambda idx=i, g=group: group_results.__setitem__(idx, run_group(g))
        )
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # Process results
    for i, (results, input_ids) in enumerate(group_results):
        if results is not None:
            logger.info(f"Group {i} completed successfully")
            valid_results = [r for r in results if r is not None]
            if valid_results:
                all_logits = torch.cat([result[0] for result in valid_results], dim=0)
                logger.info(f"Group {i} final logits shape: {all_logits.shape}")
            else:
                logger.error(f"No valid results for group {i}")
        else:
            logger.error(f"Group {i} failed")

if __name__ == '__main__':
    main()