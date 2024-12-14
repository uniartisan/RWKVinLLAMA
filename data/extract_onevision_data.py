data_path = '/home/yueyulin/data/lmms-lab/LLaVA-OneVision-Data/CLEVR-Math(MathV360K)'
import os
from datasets import load_dataset
from tqdm import tqdm
import json

data = load_dataset(data_path, split="train")

image_folder = "/home/yueyulin/data/OneVision_Data/CLEVR-Math(MathV360K)"
os.makedirs(image_folder, exist_ok=True)

converted_data = []

for da in tqdm(data):
    json_data = {}
    json_data["id"] = da["id"]
    if da["image"] is not None:
        json_data["image"] = f"{da['id']}.jpg"
        da["image"].save(os.path.join(image_folder, json_data["image"]))
    json_data["conversations"] = da["conversations"]
    converted_data.append(json_data)


with open("/home/yueyulin/data/OneVision_Data/CLEVR-Math(MathV360K)/OneVision_Data.json", "w") as f:
    json.dump(converted_data, f, indent=4, ensure_ascii=False)

