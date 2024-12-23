'''
import torch
import json
import os
import sys
import argparse
if __name__ == '__main__':

    This utility will take a model paramters file and split it into multiple files
    Args:
    --input_model_file: The model parameters file
    --output_dir: The directory to save the split files
    --num_splits: The number of splits
    The utility will split the model parameters into num_splits files and save them in output_dir
    A file named "model_params.json" will be saved in output_dir containing the information about the split files
    The splitted pattern is as follows:
    MODEL_BASE_NAME_import torch
'''
import json
import os
import sys
import argparse
from typing import Dict, List, Tuple
import numpy as np
import torch

def calculate_tensor_size(tensor: torch.Tensor) -> int:
    """Calculate the memory size of a tensor in bytes."""
    return tensor.element_size() * tensor.nelement()

def calculate_param_sizes(model: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Calculate the size of each parameter in the model."""
    return {key: calculate_tensor_size(tensor) for key, tensor in model.items()}

def distribute_params(param_sizes: Dict[str, int], num_splits: int) -> List[Dict[str, torch.Tensor]]:
    """
    Distribute parameters into splits trying to maintain equal sizes.
    Uses a greedy approach similar to the multiway number partitioning algorithm.
    """
    # Sort parameters by size in descending order
    sorted_params = sorted(param_sizes.items(), key=lambda x: x[1], reverse=True)
    
    # Initialize splits with empty dictionaries and their total sizes
    splits = [[] for _ in range(num_splits)]
    split_sizes = [0] * num_splits
    
    # Distribute parameters using greedy approach
    for param_name, param_size in sorted_params:
        # Find the split with minimum total size
        min_split_idx = split_sizes.index(min(split_sizes))
        splits[min_split_idx].append(param_name)
        split_sizes[min_split_idx] += param_size
    
    return splits

def main():
    parser = argparse.ArgumentParser(description='Split model parameters into balanced files')
    parser.add_argument('--input_model_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_splits', type=int, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load model
    print(f"Loading model from {args.input_model_file}")
    model = torch.load(args.input_model_file)
    
    # Calculate sizes of all parameters
    print("Calculating parameter sizes...")
    param_sizes = calculate_param_sizes(model)
    total_size = sum(param_sizes.values())
    
    # Distribute parameters into splits
    print(f"Distributing parameters into {args.num_splits} splits...")
    split_params = distribute_params(param_sizes, args.num_splits)
    
    # Save splits and track parameter locations
    model_params = {}  # Maps parameter names to their file locations
    model_base_name = os.path.basename(args.input_model_file).split('.')[0]
    
    for i, split_param_names in enumerate(split_params):
        split_file = os.path.join(args.output_dir, f"{model_base_name}_{i:04d}_Of_{args.num_splits}.pt")
        
        # Create split model dictionary
        split_model = {name: model[name] for name in split_param_names}
        split_size = sum(calculate_tensor_size(tensor) for tensor in split_model.values())
        
        # Save split
        print(f"Saving split {i+1}/{args.num_splits}")
        print(f"Split size: {split_size/1024/1024:.2f}MB "
              f"({split_size/total_size*100:.2f}% of total)")
        torch.save(split_model, split_file)
        
        # Record parameter locations
        for param_name in split_param_names:
            model_params[param_name] = split_file
    
    # Save parameter mapping
    model_params_file = os.path.join(args.output_dir, "model_params.json")
    with open(model_params_file, 'w') as f:
        json.dump(model_params, f, indent=4, ensure_ascii=False)
    
    print("Done!")

if __name__ == '__main__':
    main()
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_splits', type=int, required=True)
    args = parser.parse_args()
    input_model_file = args.input_model_file
    output_dir = args.output_dir
    num_splits = args.num_splits
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = torch.load(input_model_file)
    model_keys = model.keys()
    model_base_name = os.path.basename(input_model_file)
    model_base_name = model_base_name.split('.')[0]
    split_size = len(model_keys) // num_splits
    split_files = []
    model_params = {}
    for i in range(num_splits):
        print(f"Splitting {i} of {num_splits}")
        split_file = os.path.join(output_dir, f"{model_base_name}_{i:04d}_Of_{num_splits}.pt")
        split_files.append(split_file)
        split_model = {}
        for j in range(i * split_size, (i + 1) * split_size):
            key = list(model.keys())[j]
            split_model[key] = model[key]
            model_params[key] = split_file
        print(f'Saving {len(split_model.keys())} keys to {split_file}')
        torch.save(split_model, split_file)
    model_params_file = os.path.join(output_dir, "model_params.json")
    with open(model_params_file, 'w') as f:
        json.dump(model_params, f, indent=4,ensure_ascii=False)
        
        
'''
