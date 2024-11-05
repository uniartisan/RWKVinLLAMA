import torch
model_path = '/home/yueyulin/models/Qwen2.5-32B-Instruct/'
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(model_path,device_map="auto",attn_implementation="flash_attention_2", low_cpu_mem_usage=True, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_path)
print(model.hf_device_map)
input_text = "请续写悬疑小说。在一个小镇上，有一个小男孩，他的名字叫小明。"
response_text = """在一座被群山环绕的宁静小镇上，住着一个名叫小明的十岁小男孩。小明有着一头乌黑的短发和一双闪烁好奇光芒的大眼睛。他住在镇子边缘的一座老式木屋里，这间屋子不仅充满了家庭的温馨，还隐藏着许多不为人知的秘密。

一天傍晚，当夕阳的余晖染红了半边天空时，小明在自家后院的草丛中发现了一本古老的日记本。封面已经有些破损，上面布满了尘土。出于好奇，小明轻轻翻开扉页，发现里面记录了一些令人费解的文字和符号。这些日记出自一个小镇前任居民之手，年代可以追溯到一个多世纪前。

随着夜幕降临，小镇四周的树林里开始传来奇怪的声音。小明感到一阵寒意爬上脊背，但手中的日记仿佛有一种魔力，牵引着他继续阅读下去。根据日记中的线索，小明得知这个镇子曾经发生过一系列未解之谜，包括神秘失踪的村民、深夜里闪烁的光点以及无法解释的现象。而这一切似乎都与自己家族的历史息息相关。

从那晚开始，小明决定揭开隐藏在小镇历史背后的秘密。他开始调查并逐渐发现了更多令人震惊的事实：小镇上流传着一种古老的传说，关于一个能够带来好运也有可能引发灾难的力量。随着时间推移，小明意识到自己可能是解开这一切的关键人物。

在这个过程中，小明结识了一些朋友，他们一起探索小镇的秘密，面对各种未知的挑战。与此同时，也有一些不明身份的人物开始注意到了小明的行为，试图阻止他继续调查。一场围绕着古老秘密与现代勇气之间的较量悄然上演。

小明的故事就这样展开，在一次次惊心动魄的冒险中，他不仅要面对外在的危险，还要克服内心的恐惧和疑惑，寻找真相的路上，他不断成长，最终揭晓了那些古老秘密背后的真相。
"""
conversation = []
conversation.append({'role':'user','content':input_text})
conversation.append({'role':'assistant','content':response_text})
input_text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt")
inputs_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
max_len = 2048
#expand the input_ids and attention_mask to max_len
inputs_ids = torch.cat([inputs_ids,torch.zeros((1,max_len-inputs_ids.shape[1]),dtype=inputs_ids.dtype)],dim=1)
attention_mask = torch.cat([attention_mask,torch.zeros((1,max_len-attention_mask.shape[1]),dtype=attention_mask.dtype)],dim=1)
import time
start = time.time()
with torch.no_grad():
    outputs = model(input_ids=inputs_ids,attention_mask=attention_mask,output_hidden_states=True)
end = time.time()
print(f"Time elapsed: {end-start}")
print(outputs.logits.shape)
print(len(outputs.hidden_states))