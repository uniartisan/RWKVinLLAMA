import datasets
import argparse
from datasets import load_from_disk

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/rwkv/data/ultrachat')
    parser.add_argument('--min_len', type=int, default=0)
    parser.add_argument('--max_len', type=int, default=1000)
    args = parser.parse_args()
    return args

def main():
    # Parse arguments
    args = parse_args()
    datasets = load_from_disk(args.data_dir)
    #assert there is a length in the dataset
    print(datasets)
    filterd_ds = datasets.filter(lambda x: args.min_len <= x['length'] <= args.max_len,
                                 num_proc=16)
    print(filterd_ds)
if __name__ == '__main__':
    main()