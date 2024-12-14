model_path = '/home/yueyulin/models/Qwen2.5-7B-Instruct/'
from transformers import AutoModelForCausalLM
import torch
# Load model
model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16
            )
model.eval()
model.to('cuda:0')
import os
configuration_file = os.path.join(model_path, 'config.json')
with open(configuration_file, 'r') as f:
    import json
    config = json.load(f)
vocab_size = config['vocab_size']
hidden_size = config['hidden_size']
num_hidden_layers = config['num_hidden_layers']
# Generate random input ids
batch_size = 4
max_length = 128
input_ids = torch.randint(
            0, vocab_size, 
            (batch_size, max_length), 
            device='cuda:0'
        )
attention_mask = torch.ne(input_ids, 151645).to('cuda:0')
with torch.no_grad():
    batch_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False)
with torch.no_grad():
    serial_outputs = []
    for i in range(0,batch_size,2):
        serial_outputs.append(model(
                    input_ids=input_ids[i:i+2],
                    attention_mask=attention_mask[i:i+2],
                    output_hidden_states=True,
                    use_cache=False
                ))

#compare logits
bs = 2
for i in range(2):
    print(batch_outputs.logits[i*bs:(i+1)*bs])
    print('----------------')
    print(serial_outputs[i].logits)
    print(torch.max(torch.abs(batch_outputs.logits[i*bs:(i+1)*bs] - serial_outputs[i].logits)))
    print('================')