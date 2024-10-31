import datasets
import argparse
from datasets import load_from_disk

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/rwkv/data/ultrachat')
    args = parser.parse_args()
    return args

def main():
    # Parse arguments
    args = parse_args()
    datasets = load_from_disk(args.data_dir)
    #assert there is a length in the dataset
    print(datasets)
    length_data = datasets['length']
    #print the max,min,mean and std of the length of values
    import numpy as np
    print(f"Max: {np.max(length_data)}, Min: {np.min(length_data)}, Mean: {np.mean(length_data)}, Std: {np.std(length_data)}")
    #histogram of the length of values
    import matplotlib.pyplot as plt
    plt.hist(length_data, bins=100)
    #save the histogram
    plt.savefig('histogram.png')
if __name__ == '__main__':
    main()