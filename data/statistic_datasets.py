import datasets
import argparse
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=str, default='microsoft/DialoGPT-medium')
    parser.add_argument('--input_dir', type=str, default='data/ultrachat_pseudo_labels_qwen/')
    parser.add_argument('--max_len', type=int, default=2048)
    args = parser.parse_args()
    return args