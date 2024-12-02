import json
import os
import time
import yaml
import glob
import webdataset as wds
from PIL import Image, ImageFile
import jsonlines
import copy

from tqdm import tqdm

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--wds-path', type=str, default=None, help="file path", required=True)
  parser.add_argument('--output-path', type=str, default="", help="file path", required=True)
  parser.add_argument('--output-prefix', type=str, default="", help="file path", required=True)
  args = parser.parse_args()

  output = args.output_path
  if not os.path.exists(output):
      os.makedirs(output)
  else:
      print(f"Dir: {output} already existed.")

  tar_files = glob.glob(args.wds_path)
  if not tar_files:
    print(f"No files found matching the pattern: {args.wds_path}")
    exit(1)

  ## Allowed fields and Rename
  fields_mapping = dict()
  fields_mapping['id'] = 'id'
  fields_mapping['source'] = 'source'
  fields_mapping['conversations'] = 'conversations'
  fields_mapping['image'] = 'image'
  fields_mapping['tags'] = 'ram++_tags'
  fields_mapping['score'] = 'ram++_tags_score'
  fields_mapping['phash'] = 'phash'
  fields_mapping = {v: k for k, v in fields_mapping.items()}

  json_list = []
#   dataset = wds.WebDataset(args.wds_path)
  dataset = wds.WebDataset(tar_files)
  filtered = 0
  batch_size = 1000
  lines = 0
  for sample in tqdm(dataset):
    entry = copy.deepcopy(json.loads(sample['json']))
    if 'source' in entry:
      del entry['source']
    if 'ram++_tags' in entry:
      del entry['ram++_tags']
    if 'ram++_tags_score' in entry:
      del entry['ram++_tags_score']
    if 'phash' in entry:
      del entry['phash']

    img_data = sample['jpg']
    if img_data == bytes():
      pass
    else:
      file_name_without_ext, file_extension = os.path.splitext(entry['image'])
      img_filename = f"{sample['__key__']}{file_extension}"
      
      try:
        target_dir = os.path.join(output, f"{int(lines/batch_size):05d}")
        os.makedirs(target_dir, exist_ok=True)
        img_file = open(os.path.join(target_dir, img_filename), 'wb')
        img_file.write(img_data)
        img_file.close()

      except Exception as exn:
        print(exn)
        import traceback
        traceback.print_exc()
        filtered += 1
        continue
      entry['image'] = os.path.join(os.path.abspath(target_dir), img_filename)
    json_list.append(entry)
    lines += 1
    # writer.write(entry)

  json_file = os.path.join(output, f"{args.output_prefix}.json")
  with open(json_file, 'w', encoding='utf-8') as f:
      json.dump(json_list, f, ensure_ascii=False, indent=4)
  print(f"Filtered {filtered} samples.", flush=True)