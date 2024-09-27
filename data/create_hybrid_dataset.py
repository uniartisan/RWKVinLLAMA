import argparse
from transformers import AutoTokenizer
import glob
import json
from datasets import Dataset

def custom_chat_template_llama(messages):
    template = ""
    for msg in messages:
        if msg["role"] == "user":
            template += f"<|start_header_id|>user<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
        elif msg["role"] == "assistant":
            template += f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
    # Add the final assistant prompt
    template += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return template

def custom_chat_template_qwen(messages):
    template = ""
    for msg in messages:
        if msg["role"] == "user":
            template += f"<|im_start|>user\n\n{msg['content']}<|im_end|>"
        elif msg["role"] == "assistant":
            template += f"<|im_start|>assistant\n\n{msg['content']}<|im_end|>"
    # Add the final assistant prompt
    template += "<|im_start|>assistant\n\n"
    return template



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=str, default='/home/yueyulin/models/Qwen2.5-7B-Instruct/')
    parser.add_argument('--input_dir', type=str, default='/home/yueyulin/data/ultrachat_pseudo_labels_qwen/')
    parser.add_argument('--max_len', type=int, default=2048)
    parser.add_argument('--output_dir', type=str, default='/home/yueyulin/data/ultrachat_hybrid_ds/')
    args = parser.parse_args()
    return args

def create_user_input(input_text, is_llama):
    if is_llama:
        return f"<|start_header_id|>user<|end_header_id|>\n\n{input_text}<|eot_id|>"
    else:
        return f"<|im_start|>user\n{input_text}<|im_end|>\n"
def create_assistant_input(input_text, is_llama):
    if is_llama:
        return f"<|start_header_id|>assistant<|end_header_id|>\n\n{input_text}<|eot_id|>"
    else:
        return f"<|im_start|>assistant\n{input_text}<|im_end|>\n"
    
def create_inputs_labels(conversations, tokenizer, is_llama, max_len):
    # converted_str = tokenizer.apply_chat_template(conversations, tokenize=False)
    # print(converted_str)
    # json_str = json.dumps({"text": converted_str})
    # print(json_str)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    input_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    labels = []
    for conversation in conversations:
        is_user = conversation['role'] == 'user'
        input_text = conversation['content']
        if is_user:
            input_text = create_user_input(input_text, is_llama)
        else:
            input_text = create_assistant_input(input_text, is_llama)
        encoded_ids = tokenizer.encode(input_text, add_special_tokens=False)
        if is_user:
            labels.extend([-100]*len(encoded_ids))
        else:
            labels.extend(encoded_ids)
        input_ids.extend(encoded_ids)
    if tokenizer.bos_token_id is not None:
        # If the input_ids doesn't add bos_token_id, the labels will be shifted by 1
        #otherwise the labels is already shifted by 1 relative to input_ids
        labels = labels[1:]
        labels.append(-100)
    #truncate and pad
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
    else:
        input_ids = input_ids + [tokenizer.pad_token_id] * (max_len - len(input_ids))
        labels = labels + [-100] * (max_len - len(labels))
    assert len(input_ids) == len(labels)
    assert len(input_ids) == max_len
    return input_ids, labels
def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    is_llama = 'llama' in args.tokenizer.lower()
    input_dir = args.input_dir
    output_dir = args.output_dir
    max_len = args.max_len
    print(tokenizer)

    jsonl_files = glob.glob(f'{input_dir}/*.jsonl')
    dict_data = {
        'input_ids': [],
        'labels': []
    }
    for jsonl_file in jsonl_files:
        print(jsonl_file)
        with open(jsonl_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                if 'data' in data:
                    try:
                        conversations = data['data']
                        input_ids, labels = create_inputs_labels(conversations, tokenizer, is_llama, max_len)
                        dict_data['input_ids'].append(input_ids)
                        dict_data['labels'].append(labels)
                    except Exception as e:
                        print(e)
                        continue
        print(f'finished one file with {len(dict_data["input_ids"])} samples')
    ds = Dataset.from_dict(dict_data)
    ds.save_to_disk(output_dir)
    print(f'saved to {output_dir} with {len(ds)} samples')
if __name__ == '__main__':
    main()