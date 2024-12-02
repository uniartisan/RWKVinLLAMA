import torch
import torch.nn.functional as F
import logging
import cupy as cp
from cupy.cuda import nccl
import json
import torch
from torch.optim import Adam
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from server.nccl_client import InferenceClient
from profiler import time_function


def initialize_nccl_client(args):
    if not args.is_sft and args.teacher_client_mode:
        
        rank = torch.cuda.current_device()
        logging.info(f'开始初始化NCCL客户端: rank {rank}')
        # world_size = args.world_size
        cp.cuda.Device(rank).use()
        # Create student client
        groups = args.groups
        for group in groups:
            if rank in group['cuda_devices']:
                world_size = group['num_groups']
                global_rank = group['global_ranks'][group['cuda_devices'].index(rank)]
                batch_size = args.micro_bsz
                num_hidden_layers = args.n_layer
                hidden_size = args.n_embd
                max_length = args.max_seq_length
                vocab_size = args.vocab_size
                nccl_file = group['nccl_file']
                break
        with open(nccl_file,'r') as f:
            import json 
            nccl_id = json.load(f)['nccl_ids']
            nccl_id = tuple(nccl_id)
        import os
        process_id = os.getpid()
        print(f'PID:{process_id} rank {rank} is initializing student client, world_size is {world_size}, global_rank is {global_rank} with nccl_id {nccl_id}')
        client = InferenceClient(
            world_size = world_size,
            global_rank = global_rank,
            local_rank = rank,
            nccl_id = nccl_id,
            batch_size = batch_size,
            length = max_length,
            vocab_size = vocab_size,
            num_layers = num_hidden_layers,
            hidden_size = hidden_size,
            output_hidden_states = args.is_hidden_align
        )

        logging.info('NCCL客户端初始化完成')
        return client
    
@time_function
def train_step_vl(model, batch, args, teacher_engine=None, tokenizer=None):
    images = batch['images']
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    image_sizes = batch['image_sizes']

    if not args.is_sft:
        teacher_logits, teacher_hidden_states, teacher_loss = get_teacher_outputs_vl(teacher_engine, input_ids, attention_mask,labels,image_sizes,images, args)
        
        student_outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels,images=images,image_sizes=image_sizes, use_cache=False, output_hidden_states=args.is_hidden_align)
        
        if not args.is_hidden_align:
            loss, kl_loss, student_cross_entropy_loss = compute_kl_loss(student_outputs, teacher_logits, labels, args)
        else:
            kl_loss = None
            student_cross_entropy_loss = None
            loss = compute_hidden_state_loss(student_outputs, teacher_hidden_states)
        
        return loss, teacher_loss, kl_loss, student_cross_entropy_loss
    else:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,images=images,image_sizes=image_sizes, use_cache=False)
        return outputs.loss, None, None, None
@time_function
def get_teacher_outputs_vl(teacher_model, 
                           input_ids, 
                           attention_mask, 
                           labels, 
                           image_sizes,
                           images,
                           args):
    # device = input_ids.device
    
    # # 将teacher模型移动到GPU
    # teacher_model.to(device)
    
    with torch.no_grad():
        teacher_outputs = teacher_model.forward(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    images=images,
                                    labels=labels,
                                    image_sizes=image_sizes,
                                    use_cache=False,
                                    output_hidden_states=args.is_hidden_align)
    
    teacher_logits = teacher_outputs.logits
    teacher_hidden_states = teacher_outputs.hidden_states if args.is_hidden_align else None
    teacher_loss = teacher_outputs.loss
    if teacher_hidden_states is not None:
        teacher_hidden_states = torch.cat(teacher_hidden_states, dim=0)
    # 将teacher模型移回CPU
    # teacher_model.to('cpu')
    return teacher_logits, teacher_hidden_states, teacher_loss

@time_function
def train_step(model, batch, args, teacher_engine=None, tokenizer=None):
    input_ids = batch['input_ids']
    labels = batch['labels']
    attention_mask = torch.ne(input_ids, tokenizer.pad_token_id).to(input_ids.device)

    if not args.is_sft:
        if args.teacher_client_mode:
            teacher_loss = None
            teacher_logits, teacher_hidden_states = get_teacher_outputs_client_mode(model, input_ids, args)
        else:
            teacher_logits, teacher_hidden_states, teacher_loss = get_teacher_outputs(teacher_engine, input_ids, attention_mask, labels, args)
        
        student_outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False, output_hidden_states=args.is_hidden_align)
        
        if not args.is_hidden_align:
            loss, kl_loss, student_cross_entropy_loss = compute_kl_loss(student_outputs, teacher_logits, labels, args)
        else:
            kl_loss = None
            student_cross_entropy_loss = None
            loss = compute_hidden_state_loss(student_outputs, teacher_hidden_states)
        
        return loss, teacher_loss, kl_loss, student_cross_entropy_loss
    else:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
        return outputs.loss, None, None, None
@time_function
def get_teacher_outputs_client_mode(model, input_ids, args):
    b, t = input_ids.shape
    logging.info(f'rank {args.local_rank} is sending input_ids to server, shape is {input_ids.shape}')
    result = model.client.forward(input_ids=input_ids)
    if args.is_hidden_align:
        logits, hidden_states = result
        return logits, hidden_states
    else:
        logits = result
        return logits,None
@time_function
def get_teacher_outputs(teacher_model, input_ids, attention_mask, labels, args):
    # device = input_ids.device
    
    # # 将teacher模型移动到GPU
    # teacher_model.to(device)
    with torch.no_grad():
        teacher_outputs = teacher_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False, output_hidden_states=args.is_hidden_align)
    teacher_logits = teacher_outputs.logits
    teacher_hidden_states = teacher_outputs.hidden_states if args.is_hidden_align else None
    teacher_loss = teacher_outputs.loss
    if teacher_hidden_states is not None:
        teacher_hidden_states = torch.cat(teacher_hidden_states, dim=0)
    # 将teacher模型移回CPU
    # teacher_model.to('cpu')
    return teacher_logits, teacher_hidden_states, teacher_loss
@time_function
def compute_kl_loss(student_outputs, teacher_logits, labels, args):
    student_logits = student_outputs.logits
    student_cross_entropy_loss = student_outputs.loss
    targets = F.softmax(teacher_logits, dim=-1)
    
    if args.is_all_labels_kl:
        kl_loss = F.kl_div(F.log_softmax(student_logits, dim=-1), targets, reduction='batchmean')
    else:
        mask = (labels != -100).float()
        log_probs_student = F.log_softmax(student_logits, dim=-1) * mask.unsqueeze(-1)
        probs_teacher = targets * mask.unsqueeze(-1)
        kl_div = F.kl_div(log_probs_student, probs_teacher, reduction='none')
        kl_div = kl_div.sum(dim=-1)
        num_valid_elements = mask.sum()
        kl_loss = kl_div.sum() / num_valid_elements
    
    loss = args.kl_weight * kl_loss + args.ce_weight * student_cross_entropy_loss
    del student_logits, teacher_logits,labels
    return loss, kl_loss,student_cross_entropy_loss
@time_function
def compute_hidden_state_loss(student_outputs, teacher_hidden_states):
    # mask = torch.ne(labels, -100).to(labels.device)
    # mask = mask.unsqueeze(1).unsqueeze(3)
    student_hidden_states = torch.cat(student_outputs.hidden_states, dim=0)
    # student_hidden_states = student_hidden_states * mask
    # teacher_hidden_states = teacher_hidden_states * mask
    diff = student_hidden_states - teacher_hidden_states
    norms = torch.linalg.vector_norm(diff, dim=-1)
    scaled_norms = norms * (student_hidden_states[0].size(-1) ** -0.5)
    loss = scaled_norms.mean()
    # loss = torch.linalg.vector_norm(teacher_hidden_states - student_hidden_states,dim=-1).mean()*(teacher_hidden_states[0].size(-1)**-0.5)
    # loss = F.mse_loss(student_hidden_states, teacher_hidden_states.to(student_hidden_states.dtype))
    return loss

def configure_optimizer(model, args):
    lr_decay = set()
    lr_1x = set()
    lr_2x = set()
    lr_3x = set()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (("_w1" in n) or ("_w2" in n)) and (args.layerwise_lr > 0):
            lr_1x.add(n)
        elif (("time_mix" in n) or ("time_maa" in n)) and (args.layerwise_lr > 0):
            lr_2x.add(n)
        elif (("time_decay" in n) or ("time_daaaa" in n)) and (args.layerwise_lr > 0):
            lr_2x.add(n)
        elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
            lr_1x.add(n)
        elif ("time_first" in n) and (args.layerwise_lr > 0):
            lr_3x.add(n)
        elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
            lr_decay.add(n)
        else:
            lr_1x.add(n)

    lr_decay = sorted(list(lr_decay))
    lr_1x = sorted(list(lr_1x))
    lr_2x = sorted(list(lr_2x))
    lr_3x = sorted(list(lr_3x))
    param_dict = {n: p for n, p in model.named_parameters()}
    
    if args.layerwise_lr > 0:
        optim_groups = [
                {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
            ]
    else:
        optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]

    if args.weight_decay > 0:
        optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]

    if args.deepspeed:
        if args.deepspeed_offload:
            optimizer = DeepSpeedCPUAdam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
        else:
            optimizer = FusedAdam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
    else:
        optimizer = Adam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps)

    return optimizer

def validation_step(model, batch, args, teacher_model=None, tokenizer=None):
    input_ids = batch['input_ids']
    labels = batch['labels']
    attention_mask = torch.ne(input_ids, tokenizer.eos_token_id).to(input_ids.device)
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
    loss = outputs.loss
    
    # 计算perplexity
    perplexity = torch.exp(loss)
    
    result = {'val_loss': loss, 'val_perplexity': perplexity}
    
    if not args.is_sft:
        if args.teacher_client_mode:
            teacher_logits, teacher_hidden_states = get_teacher_outputs_client_mode(model, input_ids, args)
        else:
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False, output_hidden_states=args.is_hidden_align)
            teacher_logits = teacher_outputs.logits
            teacher_hidden_states = teacher_outputs.hidden_states if args.is_hidden_align else None

        # 计算teacher's loss和perplexity
        teacher_logits_reshaped = teacher_logits.view(-1, teacher_logits.size(-1))
        labels_reshaped = labels.view(-1)
        teacher_loss = F.cross_entropy(teacher_logits_reshaped, labels_reshaped)
        teacher_perplexity = torch.exp(teacher_loss)

        result.update({
            'val_teacher_loss': teacher_loss,
            'val_teacher_perplexity': teacher_perplexity
        })

    return result
