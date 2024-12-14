import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import time
from datetime import datetime
from collections import defaultdict
import queue
import statistics
import torch.cuda
import signal

logging.basicConfig(
    level=os.environ.get('LOGLEVEL', 'INFO'),
    format='%(asctime)s | %(levelname)s | %(process)d | %(thread)d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('InferenceLoadTest')

class LoadTestStatistics:
    def __init__(self):
        self.lock = threading.Lock()
        self.requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.latencies = []
        self.group_latencies = defaultdict(list)
        self.last_report_time = time.time()
        self.report_interval = 60  # Report stats every minute
        
    def add_result(self, group_id, latency, success):
        with self.lock:
            self.requests += 1
            if success:
                self.successful_requests += 1
                self.latencies.append(latency)
                self.group_latencies[group_id].append(latency)
            else:
                self.failed_requests += 1
            
            current_time = time.time()
            if current_time - self.last_report_time >= self.report_interval:
                self.report_metrics()
                self.last_report_time = current_time
    
    def report_metrics(self):
        with self.lock:
            if not self.latencies:
                return
                
            logger.info("\n=== Load Test Statistics ===")
            logger.info(f"Total Requests: {self.requests}")
            logger.info(f"Successful Requests: {self.successful_requests}")
            logger.info(f"Failed Requests: {self.failed_requests}")
            logger.info(f"Success Rate: {(self.successful_requests/self.requests)*100:.2f}%")
            logger.info(f"Average Latency: {statistics.mean(self.latencies):.2f}s")
            logger.info(f"P50 Latency: {statistics.median(self.latencies):.2f}s")
            logger.info(f"P95 Latency: {statistics.quantiles(self.latencies, n=20)[-1]:.2f}s")
            logger.info("\nPer Group Statistics:")
            for group_id, latencies in self.group_latencies.items():
                logger.info(f"Group {group_id}:")
                logger.info(f"  Requests: {len(latencies)}")
                logger.info(f"  Average Latency: {statistics.mean(latencies):.2f}s")
            logger.info("========================\n")

class LoadTest:
    def __init__(self):
        self.running = False
        self.stats = LoadTestStatistics()
        self.group_threads = []
        self.group_locks = {}
        self.validation_models = {}
        self.validation_event = threading.Event()
        
    def signal_handler(self, signum, frame):
        logger.info("Received interrupt signal. Shutting down gracefully...")
        self.running = False
        self.validation_event.set()
        self.stats.report_metrics()

    def validate_group_results(self, model, input_ids, attention_mask, results, start_rank, batch):
        """Validate results for one NCCL group with reduced validation frequency"""
        try:
            if not hasattr(self, 'validation_count'):
                self.validation_count = 0
            self.validation_count += 1
            
            # Only validate every 10th request to reduce overhead
            if self.validation_count % 10 != 0:
                return True
            device = f'cuda:{start_rank}'
            model = model.to(device)
            model.eval()
            for i, result in enumerate(results):
                if result is None:
                    logger.warning(f"Skipping validation for device {i} - no results")
                    continue
                    
                thread_logits, thread_hidden_states = result
                batch_start = i * batch
                batch_end = (i + 1) * batch
                
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
                
                if logits_diff >= 1e-3 or hidden_diff >= 1e-3:
                    logger.error(f"Validation failed for device {start_rank + i}")
                    return False
                    
            return True
                
        except Exception as e:
            logger.error(f"Error during validation: {e}", exc_info=True)
            exit(1)
            return False
            
    def initialize_group(self, group_id, nccl_file, model_path):
        """Initialize model and NCCL communicator for a group"""
        self.group_locks[group_id] = threading.Lock()
        
        # Load NCCL ID
        with open(nccl_file, 'r') as f:
            nccl_id = tuple(json.load(f)['nccl_ids'])
            
        # Initialize model for validation
        if group_id not in self.validation_models:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation='flash_attention_2'
            )
            self.validation_models[group_id] = model
        
        return nccl_id
        
    def run_continuous_group_inference(self, group_id, nccl_file, params, input_texts, start_rank):
        """Continuously run inference for one NCCL group"""
        try:
            with self.group_locks[group_id]:
                # Initialize tokenizer and get NCCL ID
                tokenizer = AutoTokenizer.from_pretrained(params['model_path'])
                nccl_id = self.initialize_group(group_id, nccl_file, params['model_path'])
                
                # Initialize clients
                clients = [None] * (params['world_size'] - 1)
                count = 0
                while self.running:
                    start_time = time.time()
                    try:
                        count += 1
                        logger.debug(f"Group {group_id} - Iteration {count}")
                        # Tokenize input for this batch
                        input_ids = tokenizer(input_texts, 
                                           return_tensors='pt',
                                           padding="max_length",
                                           max_length=params['length'],
                                           truncation=True).input_ids
                        attention_mask = torch.ne(input_ids, tokenizer.pad_token_id)
                        
                        results = [None] * (params['world_size'] - 1)
                        inference_threads = []
                        
                        def client_worker(index):
                            device_id = start_rank + index
                            
                            try:
                                if clients[index] is None:
                                    from server.nccl_client import InferenceClient
                                    clients[index] = InferenceClient(
                                        nccl_id=nccl_id,
                                        world_size=params['world_size'],
                                        batch_size=params['batch'],
                                        length=params['length'],
                                        output_hidden_states=params['output_hidden_states'],
                                        local_rank=device_id,
                                        global_rank=index + 1,
                                        vocab_size=params['vocab_size'],
                                        num_layers=params['num_layers'],
                                        hidden_size=params['hidden_size']
                                    )
                                    
                                my_input_ids = input_ids[index * params['batch']:(index + 1) * params['batch']]
                                logits, hidden_states = clients[index].forward(my_input_ids)
                                results[index] = (logits.cpu(), hidden_states.cpu())
                                logger.debug(f"Group {group_id} - Device {device_id} completed {count}")
                            except Exception as e:
                                logger.error(f"Group {group_id} - Error on device {device_id}: {e}", exc_info=True)
                                results[index] = None
                        
                        # Run inference on all devices in parallel
                        for i in range(params['world_size'] - 1):
                            thread = threading.Thread(target=client_worker, args=(i,))
                            inference_threads.append(thread)
                            thread.start()
                        
                        for thread in inference_threads:
                            thread.join()
                        logger.debug(f"Group {group_id} - All devices completed {count}")
                        # Validate results
                        # success = self.validate_group_results(
                        #     self.validation_models[group_id], 
                        #     input_ids, 
                        #     attention_mask,
                        #     results,
                        #     start_rank,
                        #     params['batch']
                        # )
                        # logger.debug(f"Group {group_id} - Iteration {count} completed in {time.time() - start_time:.2f}s, validation: {success}")
                        success = True
                        end_time = time.time()
                        
                    except Exception as e:
                        logger.error(f"Error in group {group_id} iteration: {e}", exc_info=True)
                        self.stats.add_result(group_id, time.time() - start_time, False)
                    
                    # Small delay between iterations to prevent overwhelming system
                    time.sleep(1)
                    
        except Exception as e:
            logger.error(f"Fatal error in group {group_id}: {e}", exc_info=True)
        finally:
            torch.cuda.empty_cache()
            
    def run_load_test(self, params, groups):
        """Start the continuous load test across all groups"""
        self.running = True
        signal.signal(signal.SIGINT, self.signal_handler)
        
        try:
            # Initialize all groups first
            for group in groups:
                self.initialize_group(group['group_id'], group['nccl_file'], params['model_path'])
            
            # Start continuous inference for each group
            for group in groups:
                thread = threading.Thread(
                    target=self.run_continuous_group_inference,
                    args=(
                        group['group_id'],
                        group['nccl_file'],
                        params,
                        group['input_texts'],
                        group['start_rank']
                    )
                )
                thread.daemon = True
                self.group_threads.append(thread)
            
            # Start all threads after initialization
            for thread in self.group_threads:
                thread.start()
            
            # Wait for interrupt signal
            while self.running:
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in load test: {e}", exc_info=True)
        finally:
            self.running = False
            self.validation_event.set()
            for thread in self.group_threads:
                thread.join()
            
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
        'length': 1024,
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
    
    load_test = LoadTest()
    load_test.run_load_test(params, groups)

if __name__ == '__main__':
    main()