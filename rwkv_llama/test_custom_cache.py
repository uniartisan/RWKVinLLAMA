import sys
import os

# 获取要添加的目录路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 打印要添加的目录
print(f"正在添加以下路径到 sys.path: {project_root}")

# 添加项目根目录到 sys.path
sys.path.append(project_root)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rwkv_llama.utilities import HybridCache

# 检查CUDA是否可用
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 设置模型路径
model_id = "/home/yueyulin/model/llama-3.1-8B-Instruct/"

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # 使用FP16精度
    device_map="auto"  # 自动选择可用的GPU
)

# 创建HybridCache实例
cache = HybridCache()

# 准备输入
input_text = "User: 请为我编写一个 Python 程序，输入两个数字，计算他们每个数字和集合平均值的差值。\nAssistant: 好的，"
input_ids = tokenizer(input_text, return_tensors="pt").to(device)
print(input_ids)

# 使用模型生成输出,同时使用HybridCache
with torch.no_grad():
    output = model.generate(
        input_ids = input_ids['input_ids'],
        attention_mask = input_ids['attention_mask'],
        max_length=500,
        num_return_sequences=1,
        past_key_values=cache,
        use_cache=True
    )

# 解码输出
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("生成的文本:")
print(generated_text)