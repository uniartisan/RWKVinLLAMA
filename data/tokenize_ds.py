import datasets
import os
import argparse
from transformers import AutoTokenizer
parser = argparse.ArgumentParser()
parser.add_argument('--tokenizer', type=str, default='/home/yueyulin/models/Qwen/Qwen2.5-32B-Instruct/')
parser.add_argument('--input_dir', type=str, default='/home/yueyulin/data/dclm-10B')
parser.add_argument('--max_len', type=int, default=16*1024)
parser.add_argument('--output_dir', type=str, default='/home/yueyulin/data/dclm-10B-tokenized')
args = parser.parse_args()
parquet_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.parquet')]
print(parquet_files)
tokenizer = None
def tokenize_function(examples,tokenizer_path):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    data = tokenizer(examples['text'], max_length=args.max_len, truncation=True,return_attention_mask=False,return_token_type_ids=False,return_length=True)
    labels = []
    for input_ids in data['input_ids']:
        label = input_ids[1:]
        label.append(tokenizer.pad_token_id)
        labels.append(label)
    data['labels'] = labels
    return data
original_ds = datasets.load_dataset('parquet', data_files=parquet_files)['train']
tokenized_ds = original_ds.map(lambda x: tokenize_function(x,args.tokenizer), batched=True, remove_columns=original_ds.column_names, num_proc=16)
print(tokenized_ds)
print(tokenized_ds[0])
tokenized_ds.save_to_disk(args.output_dir)