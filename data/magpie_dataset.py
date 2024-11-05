import argparse
import datasets
import glob
import os
import json
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/home/yueyulin/data/Magpie-Qwen2.5-Pro-1M-v0.1/",help="Path to the input directory")
    parser.add_argument("--model_id", help="Model ID for AutoTokenizer")
    parser.add_argument("--output_dir", help="Path to the output directory",type=str,default="/home/yueyulin/data/Magpie-Qwen2.5-Pro-1M-v0.1-Processed-JSONL/")
    args = parser.parse_args()
    parquet_files = glob.glob(os.path.join(args.input_dir, "**/*.parquet"))+glob.glob(os.path.join(args.input_dir, "*.parquet"))
    print(f'All parquet files under {args.input_dir} and its subdirectories: {parquet_files}')
    ds = datasets.load_dataset('parquet', data_files=parquet_files)['train']
    print(ds)
    #calculate the statistics of instruct_reward
    instruct_reward = ds['instruct_reward']
    import numpy as np
    instruct_reward = np.array(instruct_reward)
    print(f"Mean of instruct_reward: {np.mean(instruct_reward)}")
    print(f"Standard deviation of instruct_reward: {np.std(instruct_reward)}")
    print(f"Max of instruct_reward: {np.max(instruct_reward)}")
    print(f"Min of instruct_reward: {np.min(instruct_reward)}")
    print(f"Median of instruct_reward: {np.median(instruct_reward)}")
    file_index = 0
    max_lines = 100000
    #output the histogram of instruct_reward
    output_png = 'instruct_reward.png'
    import matplotlib.pyplot as plt
    plt.hist(instruct_reward, bins=100)
    plt.savefig(output_png)
    min_instruct_reward = np.median(instruct_reward)-np.std(instruct_reward)
    min_instruct_reward = 0 if min_instruct_reward<0 else min_instruct_reward
    max_instruct_reward = np.median(instruct_reward)+np.std(instruct_reward)
    max_instruct_reward = np.max(instruct_reward) if max_instruct_reward>np.max(instruct_reward) else max_instruct_reward
    #calculate the record counts between min_instruct_reward and max_instruct_reward
    count = 0
    for i in range(len(instruct_reward)):
        if instruct_reward[i]>=min_instruct_reward and instruct_reward[i]<=max_instruct_reward:
            count+=1
    print(f"Record counts between {min_instruct_reward} and {max_instruct_reward}: {count}")

    os.makedirs(args.output_dir, exist_ok=True)
    #output the dataset to jsonl files
    conversations = ds['conversations']
    output_file = os.path.join(args.output_dir, f'{file_index}.jsonl')
    output_file_fp = open(output_file, 'w', encoding='utf-8')
    count = 1
    for i in range(len(conversations)):
        reward = instruct_reward[i]
        if reward<min_instruct_reward or reward>max_instruct_reward:
            continue
        count += 1
        conversation = conversations[i]
        new_conversation = []
        for d in conversation:
            data = {'role':'user' if d['from']=='human' else 'assistant','content':d['value']}
            new_conversation.append(data)    
        if count%max_lines==0:
            file_index+=1
            output_file = os.path.join(args.output_dir, f'{file_index}.jsonl')
            if output_file_fp is not None:
                output_file_fp.close()
            output_file_fp = open(output_file, 'w', encoding='utf-8')
            print(f'Open file {output_file} for writing')
        output_file_fp.write(json.dumps({'data': new_conversation},ensure_ascii=False)+'\n')
    if count % max_lines != 0:
        output_file_fp.close()


if __name__ == '__main__':
    main()