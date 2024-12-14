import torch
all_logits = torch.load('all_logits.pt')
all_hidden_states = torch.load('tmp_hidden_states.pt').to(dtype=torch.float32)
logits = torch.load('pytorch_logits.pt')
hidden_states = torch.load('pytorch_hidden_states.pt')

print(f"all_logits shape: {all_logits.shape} dtype {all_logits.dtype}")
print(f"all_hidden_states shape: {all_hidden_states.shape} dtype {all_hidden_states.dtype}")
print(f"logits shape: {logits.shape} dtype {logits.dtype}")
print(f"hidden_states shape: {hidden_states.shape} dtype {hidden_states.dtype}")

#print max diff
print(f"Max difference in logits: {torch.max(torch.abs(all_logits - logits))}")
print(f"Max difference in hidden states: {torch.max(torch.abs(all_hidden_states - hidden_states))}")

#found the max difference which row between all_logits and logits
for i in range(all_hidden_states.shape[0]):
    batch_hidden_states = all_hidden_states[i]
    for j in range(batch_hidden_states.shape[0]):
        sequence_hidden_states = batch_hidden_states[j]
        for k in range(sequence_hidden_states.shape[0]):
            layer_hidden_states = sequence_hidden_states[k]
            #found the index of layer_hidden_states difference
            difference = torch.abs(layer_hidden_states - hidden_states[i,j,k])
            max_diff = torch.max(difference)
            #found the index
            if max_diff > 1e-4:
                print(f"Found the max difference in all_hidden_states and hidden_states: {max_diff}")
                print(f"all_hidden_states: {layer_hidden_states}")
                print(f"hidden_states: {hidden_states[i,j,k]}")
                break
# for i in range(all_logits.shape[0]):
#     batch_logits = all_logits[i]
#     for j in range(batch_logits.shape[0]):
#         sequence_logits = batch_logits[j]
#         for k in range(sequence_logits.shape[0]):
#             vocab_logits = sequence_logits[k]
#             #found the index of vocab_logits difference
#             difference = torch.abs(vocab_logits - logits[i,j,k])
#             max_diff = torch.max(difference)
#             #found the index
#             if max_diff > 1e-4:
#                 print(f"Found the max difference in all_logits and logits: {max_diff}")
#                 print(f"all_logits: {vocab_logits}")
#                 print(f"logits: {logits[i,j,k]}")
#                 break
