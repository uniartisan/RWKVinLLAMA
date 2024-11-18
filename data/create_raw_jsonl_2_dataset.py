'''
--input_dir input_dir --output_dir output_dir --tokenizer_path tokenizer_path --max_seq_length max_seq_length
read jsonl files in input_dir, and save the dataset to output_dir
the jsonl contains two columns: id and text. 
1. Tokenize the text column using the tokenizer initialized from tokenizer_path
2. Split the tokenized input_ids into chunks input_ids of max_seq_length
3. Create chunks of labels which has one shift of input_ids
4. padding the input_ids and labels to the same length of max_seq_length
5. record the actual length of input_ids and labels 
6. Save the dataset into output_dir, each file contains a list of dicts: {"input_ids": input_ids, "labels": labels, "length": length}
'''
import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse
import glob
import datasets
from multiprocessing import Pool
from functools import partial
import shutil
from pathlib import Path

def split_files(jsonl_files, output_tmp_dir, each_line_per_process):
    """将原始文件切分成更小的文件"""
    os.makedirs(output_tmp_dir, exist_ok=True)
    new_files = []
    current_line_count = 0
    current_file_index = 0
    current_file = None
    
    for input_file in tqdm(jsonl_files, desc="切分文件"):
        with open(input_file, 'r') as f:
            for line in f:
                if current_line_count % each_line_per_process == 0:
                    if current_file:
                        current_file.close()
                    new_file_path = os.path.join(output_tmp_dir, f'split_{current_file_index}.jsonl')
                    current_file = open(new_file_path, 'w')
                    new_files.append(new_file_path)
                    current_file_index += 1
                current_file.write(line)
                current_line_count += 1
    
    if current_file:
        current_file.close()
    
    return new_files

def process_file(file, tokenizer_path, max_seq_length, output_ds_dir):
    """处理单个文件并保存到临时数据集目录"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    file_data = {
        'input_ids': [],
        'labels': [],
        'length': []
    }
    
    print(f'处理文件 {file}')
    progress_bar = tqdm(total=os.path.getsize(file), unit='B', unit_scale=True,desc=f'处理文件 {file}')
    with open(file, 'r') as f:
        for line in f:
            item = json.loads(line)
            text = item['text']
            progress_bar.update(len(line))
            tokenized_text = tokenizer.encode(text, add_special_tokens=True)
            chunks = [tokenized_text[i:i+max_seq_length] for i in range(0, len(tokenized_text), max_seq_length)]
            
            for chunk in chunks:
                input_ids = chunk
                labels = chunk[1:] + [-100]
                length = len(chunk)
                if len(input_ids) < max_seq_length and len(input_ids) > 128:#discard the too short data
                    input_ids += [tokenizer.pad_token_id] * (max_seq_length - len(input_ids))
                    labels += [-100] * (max_seq_length - len(labels))
                file_data['input_ids'].append(input_ids)
                file_data['labels'].append(labels)
                file_data['length'].append(length)
    progress_bar.close()
    # 保存到临时数据集目录
    output_path = os.path.join(output_ds_dir, Path(file).stem)
    ds = datasets.Dataset.from_dict(file_data)
    ds.save_to_disk(output_path)
    return output_path

def merge_datasets(ds_paths, output_dir):
    """合并所有临时数据集"""
    from datasets import concatenate_datasets
    ds_list = [datasets.load_from_disk(path) for path in ds_paths]
    combined_ds = concatenate_datasets(ds_list)
    combined_ds.save_to_disk(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, required=True)
    parser.add_argument("--each_line_per_process", type=int, default=10000,
                      help='每个临时文件包含的行数')
    args = parser.parse_args()
    
    # 创建临时目录
    tmp_jsonl_dir = os.path.join(args.output_dir, 'tmp_jsonl')
    tmp_ds_dir = os.path.join(args.output_dir, 'tmp_ds')
    os.makedirs(tmp_jsonl_dir, exist_ok=True)
    os.makedirs(tmp_ds_dir, exist_ok=True)
    
    # 1. 读取并切分文件
    jsonl_files = glob.glob(f'{args.input_dir}/*.jsonl')
    split_jsonl_files = split_files(jsonl_files, tmp_jsonl_dir, args.each_line_per_process)
    
    # 2. 使用进程池处理切分后的文件
    process_file_with_args = partial(
        process_file, 
        tokenizer_path=args.tokenizer_path,
        max_seq_length=args.max_seq_length,
        output_ds_dir=tmp_ds_dir
    )
    
    num_processes = min(os.cpu_count(), len(split_jsonl_files))
    with Pool(processes=num_processes) as pool:
        ds_paths = pool.map(process_file_with_args, split_jsonl_files)
    
    # 3. 合并所有数据集
    merge_datasets(ds_paths, args.output_dir)
    
    # 4. 清理临时目录
    print("清理临时文件...")
    shutil.rmtree(tmp_jsonl_dir)
    shutil.rmtree(tmp_ds_dir)
    
    print("处理完成！")
