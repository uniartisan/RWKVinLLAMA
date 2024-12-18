import argparse
import os

if __name__ == '__main__':
    """
    Input: list of paths to the datasets
    Output: concatenated dataset path
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,nargs='+', required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    print(args)
    import datasets
    datasets_to_be_concat = []
    for input_dir in args.input_dir:
        print(f'loading {input_dir}')
        datasets_to_be_concat.append(datasets.load_from_disk(input_dir))
    print(f'concatenating datasets {datasets_to_be_concat}')
    concatenated_ds = datasets.concatenate_datasets(datasets_to_be_concat)
    print(f'saving concatenated dataset to {args.output_dir} /{concatenated_ds}')
    concatenated_ds.save_to_disk(args.output_dir)
    print('done')
    