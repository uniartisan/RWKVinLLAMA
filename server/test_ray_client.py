import ray

# 连接到Ray集群
ret = ray.init()
print(ret)
# 获取worker的引用
#get all actor names
named_actors = ray.util.list_named_actors()
print("Named actors in the cluster:", named_actors)

worker = ray.get_actor("worker")

# 调用worker的方法
model_path = '/home/yueyulin/models/Qwen2.5-32B-Instruct/'
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
input_text = "请续写悬疑小说。在一个小镇上，有一个小男孩，他的名字叫小明。"
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs['input_ids']
result = ray.get(worker.forward.remote(input_ids))
print(result)