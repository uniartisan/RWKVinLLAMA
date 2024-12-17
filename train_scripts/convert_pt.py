import torch

def convert_model(model_path, output_path):
    print(f'remove student_attn in {model_path}')
    state_dict = torch.load(model_path, map_location='cpu')
    new_state_dict = {}
    replaced_key = 0
    for k, v in state_dict.items():
        if '.student_attn.' in k:
            new_key = k.replace('.student_attn.', '.time_mixer.')
            print(f'replace {k} with {new_key}')
            replaced_key += 1
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    del state_dict
    import gc
    gc.collect()
    print(f'save new model to {output_path} replaced {replaced_key} keys')
    torch.save(new_state_dict, output_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path to the model trained by deepspeed')
    parser.add_argument('--output_path', type=str, default='model to output finally')
    args = parser.parse_args()
    convert_model(args.model_path, args.output_path)