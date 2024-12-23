import json
import argparse
import os
import sys
import psutil

# 添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
print(f"add {project_root} to sys.path")
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from rwkv_llama.utilities import HybridCache
RWKV_VERSION = os.environ.get('RWKV_VERSION', 'v7')
if RWKV_VERSION == 'v7':
    os.environ["RWKV_MY_TESTING"]='x070'
else:
    os.environ["RWKV_MY_TESTING"]='x060'
print(f'RWKV_VERSION is {RWKV_VERSION}')
if RWKV_VERSION == 'v7':
    from rwkv_llama.hybrid_model_run_rwkv7 import create_rwkv_args, HybridModel
else:
    from rwkv_llama.hybrid_model_run import create_rwkv_args, HybridModel
from transformers.modeling_utils import no_init_weights
from transformers import GenerationConfig


def process_jsonl(input_file, output_file,model_path, ckpt_file,config_file,num_gpus=1):
    """
    Process JSONL file and convert to preference dataset format
    """
    preference_data = {
        "prompt": [],
        "chosen": [],
        "rejected": []
    }
    print(f'load model from {model_path}, ckpt_file is {ckpt_file}, config_file is {config_file}')
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    transformer_config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    args = create_rwkv_args(transformer_config, config)
    with no_init_weights():
        transformer_model = AutoModelForCausalLM.from_config(transformer_config)
        
    model = HybridModel(args, transformer_config)
    model.load_checkpoint(ckpt_file)
    model = model.to(torch.bfloat16)
    if num_gpus == 1:
        print(f'loading model to cuda:0')
        model = model.to(f'cuda:0')
    gen_config = GenerationConfig(
        max_new_tokens=256,
        stop_strings=["<|im_end|>"],
        do_sample=True,
        use_cache=True,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        min_p=0.05,
        repetition_penalty=1.1,
        no_repeat_ngram_size=4,
    )
    with open(input_file, 'r') as f:
        num_processed = 0
        for line in f:
            conversation = json.loads(line)
            if 'text' in conversation:
                conversation = conversation['text']
                prompt = conversation[0]['content']
                chosen = conversation
                #genearte rejected answer
                prompt_str = "<|im_start|>user\n" + prompt + "<|im_end|><|im_start|>assistant\n"
                input_ids = tokenizer(prompt_str, return_tensors="pt").to("cuda:0")
                input_length = input_ids.input_ids.shape[1]
                cache = HybridCache()
                with torch.no_grad():
                    output = model.model.generate(
                        input_ids=input_ids["input_ids"],
                        attention_mask=input_ids["attention_mask"],
                        generation_config=gen_config,
                        tokenizer=tokenizer,
                        output_attentions=False,
                        use_cache=True,
                        past_key_values=cache,
                    )
                generated_text = tokenizer.decode(
                    output[0, input_length:], skip_special_tokens=True
                )
                import copy
                rejected = copy.deepcopy(chosen)
                rejected[1]['content'] = generated_text
                print(f'prompt is {prompt}, chosen is {chosen}, rejected is {rejected}')
                preference_data["prompt"].append(prompt)
                preference_data["chosen"].append(chosen)
                preference_data["rejected"].append(rejected)
                print(f'processed {num_processed} examples')
    #save preference data with parquet format
    print(f'save preference data to {output_file}')
    from datasets import Dataset
    dataset = Dataset.from_dict(preference_data)
    dataset.save_to_disk(output_file)

def main():
    parser = argparse.ArgumentParser(description='Convert JSONL conversations to preference dataset')
    parser.add_argument('--input_file', help='Input JSONL file path')
    parser.add_argument('--output_file', help='Output directory to save the preference dataset')
    parser.add_argument('--model_path', type=str, help='path to the model trained by deepspeed')
    parser.add_argument('--ckpt_file', type=str, default='model to output finally')
    parser.add_argument('--config_file', type=str, help='path to the config file')
    args = parser.parse_args()
    
    process_jsonl(args.input_file, args.output_file,args.model_path, args.ckpt_file,args.config_file)    

if __name__ == "__main__":
    main()