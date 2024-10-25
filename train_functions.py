import torch
import torch.nn.functional as F
import logging
import cupy as cp
from cupy.cuda import nccl
import json
import torch
from torch.optim import Adam
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

def initialize_nccl_group(args, model):
    if not args.is_sft and args.teacher_client_mode:
        logging.info('开始初始化进程组')
        rank = args.rank
        world_size = (args.world_size // args.num_groups) + 1
        cp.cuda.Device(rank).use()
        group_id = rank // (world_size - 1)
        logging.info(f'全局rank {rank} 在组 {group_id} 中,世界大小为 {world_size}')
        nccl_file = f'{args.nccl_file}_{group_id}'
        
        with open(nccl_file, 'r') as f:
            print(f'从 {nccl_file} 加载nccl_id')
            nccl_id = json.load(f)['nccl_id']
            args.nccl_id = tuple(nccl_id)
            print("NCCL ID:", nccl_id)
        
        stream = cp.cuda.Stream(non_blocking=True)
        
        args.server_rank = world_size - 1
        rank = rank % (world_size - 1)
        recv_buffer = cp.empty((args.micro_bsz, args.max_seq_length, model.config.vocab_size), dtype=cp.float32)
        
        if args.is_hidden_align:
            teacher_hidden_states_buffer = cp.empty((args.micro_bsz * model.config.num_hidden_layers, args.max_seq_length, model.config.hidden_size), dtype=cp.float32)
        else:
            teacher_hidden_states_buffer = None

        logging.info(f'初始化进程组,本地rank为 {rank},世界大小为 {world_size}, nccl_id为 {args.nccl_id}')
        comm = nccl.NcclCommunicator(world_size, args.nccl_id, rank)
        logging.info(f'完成进程组初始化,本地rank为 {rank}')

        return comm, stream, recv_buffer, teacher_hidden_states_buffer
    
    return None, None, None, None

def train_step(model, batch, args, teacher_model=None, tokenizer=None):
    input_ids = batch['input_ids']
    labels = batch['labels']
    attention_mask = torch.ne(input_ids, tokenizer.eos_token_id).to(input_ids.device)

    if not args.is_sft:
        if args.teacher_client_mode:
            teacher_loss = None
            teacher_logits, teacher_hidden_states = get_teacher_outputs_client_mode(model, input_ids, args)
        else:
            teacher_logits, teacher_hidden_states, teacher_loss = get_teacher_outputs(teacher_model, input_ids, attention_mask, labels, args)
        
        student_outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False, output_hidden_states=args.is_hidden_align)
        
        if not args.is_hidden_align:
            loss, kl_loss, student_cross_entropy_loss = compute_kl_loss(student_outputs, teacher_logits, labels, args)
        else:
            kl_loss = None
            student_cross_entropy_loss = None
            loss = compute_hidden_state_loss(student_outputs, teacher_hidden_states, labels)
        
        return loss, teacher_loss, kl_loss, student_cross_entropy_loss
    else:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
        return outputs.loss, None, None, None

def get_teacher_outputs_client_mode(model, input_ids, args):
    b, t = input_ids.shape
    logging.info(f'rank {args.rank} is sending input_ids to server, shape is {input_ids.shape}')
    model.comm.send(input_ids.data_ptr(), input_ids.size(0)*input_ids.size(1), model.nccl.NCCL_INT64, args.server_rank, model.stream.ptr)
    model.stream.synchronize()
    
    logging.info(f'rank {args.rank} is receiving teacher_logits from server')
    model.comm.recv(model.recv_buffer.data.ptr, model.recv_buffer.size, model.nccl.NCCL_FLOAT, args.server_rank, model.stream.ptr)
    model.stream.synchronize()
    teacher_logits = torch.as_tensor(model.recv_buffer, device=input_ids.device, dtype=torch.float32)
    
    teacher_hidden_states = None
    if args.is_hidden_align:
        logging.info(f'rank {args.rank} is receiving teacher_hidden_states from server')
        model.comm.recv(model.teacher_hidden_states_buffer.data.ptr, model.teacher_hidden_states_buffer.size, model.nccl.NCCL_FLOAT, args.server_rank, model.stream.ptr)
        model.stream.synchronize()
        teacher_hidden_states = torch.as_tensor(model.teacher_hidden_states_buffer, device=input_ids.device, dtype=torch.float32)
    
    return teacher_logits, teacher_hidden_states

def get_teacher_outputs(teacher_model, input_ids, attention_mask, labels, args):
    with torch.no_grad():
        teacher_outputs = teacher_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False, output_hidden_states=args.is_hidden_align)
    teacher_logits = teacher_outputs.logits
    teacher_hidden_states = teacher_outputs.hidden_states if args.is_hidden_align else None
    teacher_loss = teacher_outputs.loss
    if teacher_hidden_states is not None:
        teacher_hidden_states = torch.cat(teacher_hidden_states[1:], dim=0)
    return teacher_logits, teacher_hidden_states, teacher_loss

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
    return loss, kl_loss,student_cross_entropy_loss

def compute_hidden_state_loss(student_outputs, teacher_hidden_states, labels):
    mask = torch.ne(labels, -100).to(labels.device)
    mask = mask.unsqueeze(1).unsqueeze(3)
    student_hidden_states = torch.cat(student_outputs.hidden_states[1:], dim=0)
    student_hidden_states = student_hidden_states * mask
    teacher_hidden_states = teacher_hidden_states * mask
    loss = F.mse_loss(student_hidden_states, teacher_hidden_states.to(student_hidden_states.dtype))
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
            if args.my_pile_stage == 2:
                lr_2x.add(n)
            else:
                lr_1x.add(n)
        elif (("time_decay" in n) or ("time_daaaa" in n)) and (args.layerwise_lr > 0):
            if args.my_pile_stage == 2:
                lr_3x.add(n)
            else:
                lr_2x.add(n)
        elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
            if args.my_pile_stage == 2:
                lr_2x.add(n)
            else:
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
        if args.my_pile_stage == 2:
            optim_groups = [
                {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},
                {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},
            ]
        else:
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
