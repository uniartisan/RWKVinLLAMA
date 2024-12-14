import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_dir)
print(f'add path {parent_dir}')
from data.llava_data import LazySupervisedDataset, DataCollatorForSupervisedDataset,DataArguments
from torch.utils.data import DataLoader
from llava.model.builder import load_pretrained_model
data_path = '/home/yueyulin/data/MM_stage3/stage3.json'
tokenizer_path = '/home/yueyulin/models/lmms-lab/llava-onevision-qwen2-7b-si/'
pretrained = "/home/yueyulin/models/lmms-lab/llava-onevision-qwen2-7b-si"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
llava_model_args = {
    "multimodal": True,
    "attn_implementation": "flash_attention_2",
}
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, **llava_model_args)  # Add any other thing you want to pass in llava_model_args

print(tokenizer)
print(model)
print(image_processor)
print(max_length)
      
data_args = DataArguments(data_path=data_path,
image_aspect_ratio="anyres_max_9",
image_grid_pinpoints=[
    [
      384,
      384
    ],
    [
      384,
      768
    ],
    [
      384,
      1152
    ],
    [
      384,
      1536
    ],
    [
      384,
      1920
    ],
    [
      384,
      2304
    ],
    [
      768,
      384
    ],
    [
      768,
      768
    ],
    [
      768,
      1152
    ],
    [
      768,
      1536
    ],
    [
      768,
      1920
    ],
    [
      768,
      2304
    ],
    [
      1152,
      384
    ],
    [
      1152,
      768
    ],
    [
      1152,
      1152
    ],
    [
      1152,
      1536
    ],
    [
      1152,
      1920
    ],
    [
      1152,
      2304
    ],
    [
      1536,
      384
    ],
    [
      1536,
      768
    ],
    [
      1536,
      1152
    ],
    [
      1536,
      1536
    ],
    [
      1536,
      1920
    ],
    [
      1536,
      2304
    ],
    [
      1920,
      384
    ],
    [
      1920,
      768
    ],
    [
      1920,
      1152
    ],
    [
      1920,
      1536
    ],
    [
      1920,
      1920
    ],
    [
      1920,
      2304
    ],
    [
      2304,
      384
    ],
    [
      2304,
      768
    ],
    [
      2304,
      1152
    ],
    [
      2304,
      1536
    ],
    [
      2304,
      1920
    ],
    [
      2304,
      2304
    ]
  ],
  )
data_args.image_processor = image_processor
data_args.is_multimodal = True
data_args.image_folder = ''
data_args.mm_use_im_start_end = True
# train_data_module = make_supervised_data_module(tokenizer, data_path,data_args)
ds = LazySupervisedDataset(data_path, tokenizer, data_args)
collator = DataCollatorForSupervisedDataset(tokenizer)
dl = DataLoader(ds, batch_size=2, collate_fn=collator)
print(ds)
print(dl)
for d in dl:
    print(d)
    print(d['images'][0].shape)
    break