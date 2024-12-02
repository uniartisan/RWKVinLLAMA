# 1. Data Preparation

Download the BAAI/Infinity-MM to path {Infinity-MM}. 

Run the script to create data for VL_Tuner:
```bash
export INFINITY_MM={Infinity-MM}
export OUTPUT_PATH=SOME_OUTPUT_PATH
export PREFIX=SOME_PREFIX
python3 test/test_data.py --wds-path $INFINITY_MM/stage3/onevision-SI/ --output-path $OUTPUT_PATH --output-prefix $PREFIX  
```

# 2. Model Training

Run the script to replace the self attention to RWKV time mixer:

```bash
deepspeed --num_nodes 1 --num_gpus 4 train_scripts/train_hybrid_vl_deepspeed.py --deepspeed --deepspeed_offload --world_size 4 --train_batch_size 4 --micro_bsz 1 --pretrained_model /home/yueyulin/models/BAAI/Aquila-VL-2B-llava-qwen/ --data_path $OUTPUT_PATH/$PREFIX.json
```

# 3. Caustion

In the train_hybrid_vl_deepspeed.py, I set the RWKV's context length to 4096, if the input_embs(length of input_ids + length of encoded images) is larger than 4096, you need to change the context length to a larger value. 

```python
import sys
import os
import json
def setup_env():
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path.append(parent_dir)
    # rwkv_path = os.path.join(parent_dir, 'rwkv7')
    # sys.path.append(rwkv_path)
    rwkv6_path = os.path.join(parent_dir, 'rwkv')
    sys.path.append(rwkv6_path)
    rwkv_llama_path = os.path.join(parent_dir, 'rwkv_llama')
    sys.path.append(rwkv_llama_path)
    # print(f'add path: {rwkv_path} to sys.path')
    print(f'add path: {parent_dir} to sys.path')
    print(f'add path: {rwkv_llama_path} to sys.path')
    # os.environ['CUDA_HOME'] = '/usr/local/cuda-12.1'
    os.environ['RWKV_JIT_ON'] = '0'
    os.environ['RWKV_T_MAX'] = '4096'
    os.environ['RWKV_FLOAT_MODE'] = 'bf16'
    os.environ['RWKV_HEAD_SIZE_A'] = '64'
    os.environ['RWKV_T_MAX'] = '4096'
    os.environ["RWKV_MY_TESTING"]='x060'
    os.environ['RWKV_CTXLEN'] = '4096'
    if 'WKV' not in os.environ:
        os.environ['WKV'] = ''
    if "RWKV_TRAIN_TYPE" not in os.environ:
        os.environ["RWKV_TRAIN_TYPE"] = ''
    if 'RWKV_VERSION' not in os.environ:
        os.environ['RWKV_VERSION'] = 'v6'
setup_env()
```