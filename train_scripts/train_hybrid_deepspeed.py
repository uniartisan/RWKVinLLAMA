import sys
import os
def setup_env():
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path.append(parent_dir)
    rwkv_path = os.path.join(parent_dir, 'rwkv7')
    sys.path.append(rwkv_path)
    rwkv_llama_path = os.path.join(parent_dir, 'rwkv_llama')
    sys.path.append(rwkv_llama_path)
    print(f'add path: {rwkv_path} to sys.path')
    print(f'add path: {rwkv_llama_path} to sys.path')
    os.environ['CUDA_HOME'] = '/usr/local/cuda-12.1'
    os.environ['RWKV_JIT_ON'] = '0'
    os.environ['RWKV_T_MAX'] = '4096'
    os.environ['RWKV_FLOAT_MODE'] = 'bf16'
    os.environ['RWKV_HEAD_SIZE_A'] = '64'
    os.environ['RWKV_T_MAX'] = '4096'
    os.environ["RWKV_MY_TESTING"]='x060'
    os.environ['RWKV_CTXLEN'] = '4096'
    if 'WKV' not in os.environ:
        os.environ['WKV'] = ''
    if "RWKV_TRAIN_TYPE" not in os.environ:
        os.environ["RWKV_TRAIN_TYPE"] = ''
setup_env()

import argparse
import yaml
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
from hybrid_model import HybridModel
from train_functions import initialize_nccl_group, configure_optimizer, train_step, validation_step
from data.c4_datasets import load_and_interleave_c4, data_collator
from data.multi_source_datasets import load_and_interleave_datasets
import datasets
import json
import math
import time
import wandb
from tqdm import tqdm



def create_arg_parser():
    node_rank = int(os.environ.get('NODE_RANK', 0))
    num_gpus = int(os.environ.get('NUM_GPUS', 1))
    world_size = int(os.environ.get('WORLD_SIZE', 7))
    print(f'node_rank: {node_rank}, num_gpus: {num_gpus}, world_size: {world_size}')
    parser = argparse.ArgumentParser(description='MLM trainer')
    parser.add_argument('--config_file', type=str,default='configs/test_hybrid.yaml', help='training config file')
    parser.add_argument('--preprocessed_data',type=str,nargs='+',help='preprocessed data directory')
    parser.add_argument('--output_dir', type=str, default='/data/rwkv/tmp',help='directory to save the trained model')
    parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs to train the model')
    parser.add_argument('--max_seq_length', type=int, default=512, help='maximum sequence length to train the model')
    parser.add_argument('--num_devices', type=int, default = 1,help='number of devices to train the model')
    
    
    parser.add_argument('--dropout', type=float, default=0, help='dropout rate in the model')
    parser.add_argument('--grad_cp', type=int, default=0, help='gradient checkpoint in the model')
    parser.add_argument('--save_per_batches', type=int, default=10000, help='number of batches to save the model')
    parser.add_argument('--my_exit', type=int, default=300, help='exit condition in the model')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay in the model')
    parser.add_argument('--lr_init', type=float, default=6e-4, help='initial learning rate in the model')
    parser.add_argument('--lr_final', type=float, default=1e-5, help='final learning rate in the model')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 parameter in the Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.95, help='beta2 parameter in the Adam optimizer')
    parser.add_argument('--layerwise_lr', type=float, nargs='+', default=1, help='layerwise learning rate in the model')
    parser.add_argument('--adam_eps', type=float, default=1e-8, help='epsilon parameter in the Adam optimizer')
    parser.add_argument('--warmup_steps', type=int, default=50, help='warmup steps in the model')
    parser.add_argument('--epoch_begin', type=int, default=0, help='beginning epoch for the training')
    parser.add_argument('--epoch_count', type=int, default=150, help='total number of epochs for the training')
    parser.add_argument('--epoch_save', type=int, default=1, help='number of epochs after which the model is saved')
    parser.add_argument('--max_epochs', type=int, default=150, help='maximum number of epochs for the training')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='number of epochs after which the validation is checked')
    parser.add_argument('--val_check_interval', type=int, default=5000, help='number of epochs after which the validation is checked')
    parser.add_argument('--num_sanity_val_steps', type=int, default=0, help='number of validation steps for sanity check at the beginning of training')
    parser.add_argument('--log_every_n_steps', type=int, default=5000, help='number of steps after which the training progress will be logged')
    parser.add_argument('--enable_checkpointing', type=bool, default=False, help='flag to enable checkpointing')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='number of batches to accumulate before performing a backward/update pass')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='maximum gradient norm')
    parser.add_argument('--num_nodes', type=int, default=1, help='number of nodes for distributed training')
    parser.add_argument('--micro_bsz', type=int,default=2, help='micro batch size for training')
    parser.add_argument('--real_bsz', type=int, help='real batch size for training')
    parser.add_argument('--my_pile_stage', type=int, default=0, help='pile stage in the model')
    parser.add_argument('--my_pile_edecay', type=float, default=0, help='pile exponential decay in the model')
    parser.add_argument('--weight_decay_final', type=float, default=-1, help='final weight decay in the model')
    parser.add_argument('--proj_dir', type=str, help='project directory to save the model and logs')
    parser.add_argument('--eval_every_steps', type=int, default=100, help='number of steps after which the model is evaluated')
    parser.add_argument('--wandb', type=str, default='hybrid_trainer', help='wandb project name')
    parser.add_argument('--run_name', type=str, default='hybrid_trainer_a800', help='run name for wandb logging')
    parser.add_argument('--strategy', type=str, default='deepspeed_stage_2_offload', help='strategy for distributed training')
    parser.add_argument("--ds_bucket_mb", default=200, type=int)  # deepspeed bucket size in MB. 200 seems enough
    parser.add_argument('--my_qa_mask', type=int, default=0)
    parser.add_argument('--optim',type=str,default='adam',help='optimizer')
    parser.add_argument('--train_type', type=str, default='', help='train type')
    parser.add_argument('--skip_steps',type=int,default=0,help='skip steps in the peft checkpoint')

    parser.add_argument('--ckpt_file', type=str, default=None, help='checkpoint file')
    # 添加DeepSpeed相关的参数
    parser.add_argument('--deepspeed', action='store_true', help='Enable DeepSpeed')
    parser.add_argument('--deepspeed_config', type=str, default=None, help='Path to DeepSpeed config file')
    parser.add_argument('--deepspeed_stage', type=int, default=2, choices=[0, 1, 2, 3], help='DeepSpeed ZeRO stage')
    parser.add_argument('--deepspeed_offload', action='store_true', help='Enable CPU offloading')
    parser.add_argument('--train_batch_size', type=int, default=None, help='train batch size')
    return parser

def lr_schedule(args, step):
    w_step = args.warmup_steps
    if args.lr_final == args.lr_init or args.epoch_count == 0:
        return args.lr_init
    
    decay_step = step - args.my_pile_edecay * args.epoch_steps
    decay_total = (args.epoch_count - args.my_pile_edecay) * args.epoch_steps
    progress = (decay_step - w_step + 1) / (decay_total - w_step)
    progress = min(1, max(0, progress))

    if args.lr_final == 0 or args.lr_init == 0:  # linear decay
        lr = args.lr_init + (args.lr_final - args.lr_init) * progress
    else:  # exp decay
        lr = args.lr_init * math.exp(math.log(args.lr_final / args.lr_init) * pow(progress, 1))

    if step < w_step:
        lr = lr * (0.01 + 0.99 * step / w_step)
    
    return lr

def weight_decay_schedule(args, progress):
    if args.weight_decay_final > 0:
        return args.weight_decay * math.exp(math.log(args.weight_decay_final / args.weight_decay) * progress)
    return args.weight_decay

def on_train_batch_start(args, model_engine, global_step, epoch):
    real_step = global_step + args.epoch_begin * args.epoch_steps

    # LR schedule
    lr = lr_schedule(args, real_step)
    
    # Weight decay schedule
    progress = (real_step - args.warmup_steps + 1) / ((args.epoch_count - args.my_pile_edecay) * args.epoch_steps - args.warmup_steps)
    progress = min(1, max(0, progress))
    wd_now = weight_decay_schedule(args, progress)

    # 更新优化器参数
    for param_group in model_engine.optimizer.param_groups:
        if param_group["weight_decay"] > 0:
            param_group["weight_decay"] = wd_now
        if args.layerwise_lr > 0:
            param_group["lr"] = lr * param_group["my_lr_scale"]
        else:
            param_group["lr"] = lr

    # 初始化日志（仅在第一步执行）
    if global_step == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "train_log.txt"), "a") as f:
            f.write(f"NEW RUN {time.strftime('%Y-%m-%d %H:%M:%S')}\n{vars(args)}\n")

    return lr, wd_now

# 在主训练循环开始前初始化tqdm
pbar = None

def on_train_batch_end(args, batch_idx, model_engine, loss, teacher_loss, kl_loss, student_cross_entropy_loss, global_step, epoch, last_log_time, token_per_step, is_accumulation_step, pbar):
    current_time = time.time()
    elapsed_time = current_time - last_log_time
    steps_per_second = 1 / elapsed_time
    kt_s = token_per_step * steps_per_second / 1000  # K tokens per second

    # 只在实际更新参数时更新进度条
    if is_accumulation_step and model_engine.local_rank == 0:
        if pbar is None:
            pbar = tqdm(total=args.epoch_steps, desc=f"Epoch {epoch}")
        
        pbar.update(1)
        pbar.set_postfix({
            'loss': f'{loss:.4f}',
            'steps/s': f'{steps_per_second:.2f}',
            'kt/s': f'{kt_s:.2f}'
        })
        
        if args.wandb:
            wandb.log({
                "loss": loss,
                "lr": model_engine.optimizer.param_groups[0]['lr'],
                "weight_decay": model_engine.optimizer.param_groups[0]['weight_decay'],
                "steps_per_second": steps_per_second,
                "kt/s": kt_s,
                "global_step": global_step,
                "Gtokens": global_step * token_per_step*args.accumulate_grad_batches / 1e9,
                "epoch": epoch,
                "teacher_loss": teacher_loss,
                "kl_loss": kl_loss,
                "student_cross_entropy_loss": student_cross_entropy_loss,
            })

    real_step = batch_idx
    if real_step % args.save_per_batches == 0 and real_step > 0:
        pbar.write(f'Saving trainable to {args.output_dir}')
        output_dir = f"{args.output_dir}/epoch_{epoch}_step_{real_step}"
        try:
            model_engine.save_checkpoint(output_dir)
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            import traceback
            traceback.print_exc()
        print(f'saved checkpoint to {output_dir}')

    return current_time, pbar
import torch.distributed as dist
def setup_distributed():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

if __name__ == '__main__':
    parser = create_arg_parser()
    args = parser.parse_args()
    print(args)
    if args.num_nodes > 1:
        setup_distributed()
    # 加载配置
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)

    # 设置设备和数据类型
    dtype = torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型和分词器
    transformer_model = AutoModelForCausalLM.from_pretrained(config['Llama']['model_id'],
                                                            torch_dtype=dtype, device_map={'':'cpu'})
    tokenizer = AutoTokenizer.from_pretrained(config['Llama']['model_id'])
    tokenizer.pad_token = tokenizer.eos_token

    # 设置参数
    args.my_pos_emb = 0
    args.head_size_a = 64
    args.head_size_divisor = 8
    args.ctx_len = 4096
    args.n_layer = transformer_model.config.num_hidden_layers
    args.n_embd = transformer_model.config.hidden_size
    args.dim_att = transformer_model.config.hidden_size
    args.dim_ffn = transformer_model.config.intermediate_size
    args.pre_ffn = 0
    args.head_qk = 0
    args.tiny_att_dim = 0
    args.tiny_att_layer = -999
    args.vocab_size = transformer_model.config.vocab_size
    args.layers = config['RWKV']['layers']
    args.pad_id = tokenizer.eos_token_id
    args.betas = (args.beta1, args.beta2)
    args.kl_weight = config['kl_weight']
    args.ce_weight = config['ce_weight']
    args.model_file = config['model_file']
    args.real_bsz = args.micro_bsz * args.accumulate_grad_batches * args.num_devices * args.num_nodes
    args.teacher_client_mode = config['teach_mode']['is_client']
    args.nccl_file = config['teach_mode']['nccl_file']
    args.num_groups = config['teach_mode']['num_groups']
    args.is_hidden_align = config['teach_mode']['is_hidden_align']
    args.is_sft = config.get('is_sft', False)
    args.is_llama_ffn = config.get('is_llama_ffn', False)
    args.is_rwkv_att_only = config.get('is_rwkv_att_only', False)
    args.is_all_labels_kl = config.get('is_all_labels_kl', False)

    # 初始化教师模型
    if not args.teacher_client_mode:
        teacher_model = AutoModelForCausalLM.from_pretrained(config['Llama']['model_id'], torch_dtype=dtype, attn_implementation='flash_attention_2')
        teacher_model.eval()
    else:
        assert args.num_devices % args.num_groups == 0
        teacher_model = None

    # 初始化混合模型
    model = HybridModel(transformer_model, args, teacher_model, tokenizer)
    if args.ckpt_file is not None:
        dict_set = torch.load(args.ckpt_file)
        info = model.load_state_dict(dict_set, strict=False)
        print(f'load model from {args.ckpt_file}, info is {info}')
        del dict_set
    # 设置模型参数的训练状态
    if args.is_rwkv_att_only:
        print('only rwkv att is trained')
        for name, param in model.named_parameters():
            if not 'self_attn.' in name:
                param.requires_grad = False
            print(name, param.shape, param.requires_grad)
    else:
        if args.is_llama_ffn:
            print('keep llama ffn frozen')
            for name, param in model.named_parameters():
                if not 'block.' in name or 'ffn' in name:
                    param.requires_grad = False
                print(name, param.shape, param.requires_grad)
        else:
            print('keep other modules frozen except rwkv block')
            for name, param in model.named_parameters():
                if not 'block.' in name:
                    param.requires_grad = False
                print(name, param.shape, param.requires_grad)

    # 准备数据加载器
    if args.preprocessed_data is not None:
        print(f'load preprocessed data from {args.preprocessed_data}')
        from data.multi_source_datasets import data_collator
        from functools import partial
        data_collator = partial(data_collator, max_seq_length=args.max_seq_length)
        train_dir = args.preprocessed_data[0]
        val_dir = args.preprocessed_data[1] if len(args.preprocessed_data) > 1 else None
        train_ds = datasets.load_from_disk(train_dir)
        train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=args.micro_bsz, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, collate_fn=data_collator)
        if val_dir is not None:
            val_ds = datasets.load_from_disk(val_dir)
            val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=args.micro_bsz, shuffle=False, num_workers=4, pin_memory=True, drop_last=True, collate_fn=data_collator)    
        else:
            val_dataloader = None
        print(f'load preprocessed data from {args.preprocessed_data} done')
    else:
        # 处理其他数据加载情况
        pass

    # 设置DeepSpeed配置
    if args.deepspeed:
        if args.deepspeed_config:
            # 如果提供了 DeepSpeed 配置文件，直接加载它
            with open(args.deepspeed_config, 'r') as f:
                ds_config = json.load(f)
        else:
            # 否则，根据命令行参数创建配置
            ds_config = {
                "train_batch_size": args.trainq_batch_size,
                "bf16": {
                    "enabled": True
                },
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "zero_optimization": {
                    "stage": args.deepspeed_stage,
                    "stage3_max_live_parameters": 1e9,
                    "stage3_max_reuse_distance": 1e9,
                    "stage3_prefetch_bucket_size": 1e7,
                    "memory_efficient_linear": True,
                    "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": True
                    } ,
                    "allgather_partitions": True,
                    "allgather_bucket_size": args.ds_bucket_mb * 1000 * 1000,
                    "overlap_comm": True,
                    "reduce_scatter": True,
                    "reduce_bucket_size": args.ds_bucket_mb * 1000 * 1000,
                    "contiguous_gradients": True
                },
                "gradient_clipping": args.gradient_clip_val,
                "gradient_checkpointing": args.grad_cp == 1,
                "compile": {
                    "disable": False,
                    "backend": "inductor"
                },
                "zero_allow_untested_optimizer": True,
                "gradient_accumulation_steps": args.accumulate_grad_batches if args.accumulate_grad_batches > 1 else None,
                "activation_checkpointing": {
                    "partition_activations": True,
                    "cpu_checkpointing": True,
                    "contiguous_memory_optimization": True,
                    "number_checkpoints": 2
                }
            }

        # 手动配置优化器
        optimizer = configure_optimizer(model, args)

        print(f'optimizer is {optimizer}')

        # 初始化 DeepSpeed
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=ds_config
        )
    else:
        # 如果不使用 DeepSpeed，使用普通的优化器
        print('not using deepspeed, EXIT')
        exit()
    # 初始化NCCL组
    model.comm, model.stream, model.recv_buffer, model.teacher_hidden_states_buffer = initialize_nccl_group(args, model)

    # 只在主进程上初始化wandb
    if args.wandb and model_engine.local_rank == 0:
        print(f'init wandb, project is {args.wandb}, name is {args.run_name}')
        wandb.init(project=args.wandb, name=args.run_name, config=args)

    # 初始化一些变量
    args.epoch_steps = len(train_dataloader) // args.accumulate_grad_batches
    global_step = 0
    last_log_time = time.time()
    token_per_step = args.max_seq_length * args.micro_bsz * args.num_nodes * args.num_devices
    if model_engine.local_rank == 0:
        from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live
        from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
        def print_deepspeed_model_info(model_engine):
            if hasattr(model_engine, 'module'):
                model = model_engine.module
            else:
                model = model_engine

            if model_engine.zero_optimization_stage() == 0:
                print(model)
                total_params = sum(p.numel() for p in model.parameters())
                print(f'Total Parameters: {total_params:,d}')
            elif model_engine.zero_optimization_stage() in [1, 2]:
                estimate_zero2_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)
            elif model_engine.zero_optimization_stage() == 3:
                estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=4, num_nodes=1)
        print_deepspeed_model_info(model_engine)
    # 训练循环
    for epoch in range(args.max_epochs):
        model_engine.train()
        if model_engine.local_rank == 0:
            pbar = tqdm(total=args.epoch_steps, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(train_dataloader):
            lr, wd_now = on_train_batch_start(args, model_engine, global_step, epoch)

            batch = {k: v.to(model_engine.device) for k, v in batch.items()}
            
            # 前向传播
            loss, teacher_loss, kl_loss, student_cross_entropy_loss = train_step(model_engine, batch, args, teacher_model, tokenizer)
            
            # 缩放损失
            loss = loss / args.accumulate_grad_batches
            
            # 反向传播
            model_engine.backward(loss)

            is_accumulation_step = (batch_idx + 1) % args.accumulate_grad_batches == 0

            if is_accumulation_step:
                model_engine.step()
                model_engine.zero_grad()
                global_step += 1

            # 每一步都调用 on_train_batch_end，但只在累积步骤结束时更新进度条
            last_log_time, pbar = on_train_batch_end(
                args, batch_idx, model_engine, loss.item(), teacher_loss, kl_loss, student_cross_entropy_loss,
                global_step, epoch, last_log_time, token_per_step, is_accumulation_step, pbar
            )

        # 处理最后一个不完整的累积批次（如果有的话）
        if len(train_dataloader) % args.accumulate_grad_batches != 0:
            model_engine.step()
            model_engine.zero_grad()
            global_step += 1
            
            last_log_time, pbar = on_train_batch_end(
                args, batch_idx, model_engine, loss, teacher_loss, kl_loss, student_cross_entropy_loss,
                global_step, epoch, last_log_time, token_per_step, True, pbar
            )

        # 验证
        if val_dataloader:
            model_engine.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_dataloader:
                    batch = {k: v.to(model_engine.device) for k, v in batch.items()}
                    result = validation_step(model_engine, batch, args, teacher_model, tokenizer)
                    val_losses.append(result['val_loss'].item())
            
            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f"Epoch {epoch}, Validation Loss: {avg_val_loss}")

        # 保存检查点
        if args.output_dir:
            if args.deepspeed:
                model_engine.save_checkpoint(args.output_dir, f"checkpoint-epoch{epoch}")
            else:
                torch.save(model.state_dict(), os.path.join(args.output_dir, f"checkpoint-epoch{epoch}.pt"))

        if pbar is not None:
            pbar.close()

    print("Training completed")
    if args.wandb and model_engine.local_rank == 0:
        wandb.finish()






