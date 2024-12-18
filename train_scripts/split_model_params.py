import torch
import json
import os
import sys
import argparse

if __name__ == '__main__':
    '''
    This utility will take a model paramters file and split it into multiple files
    Args:
    --input_model_file: The model parameters file
    --output_dir: The directory to save the split files
    --num_splits: The number of splits
    The utility will split the model parameters into num_splits files and save them in output_dir
    A file named "model_params.json" will be saved in output_dir containing the information about the split files
    The splitted pattern is as follows:
    MODEL_BASE_NAME_0000_Of_NUM_SPLITS.pt
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
        
        
    