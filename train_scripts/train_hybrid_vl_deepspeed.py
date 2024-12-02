import sys
import os
import json
def setup_env():
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path.append(parent_dir)
    # rwkv_path = os.path.join(parent_dir, 'rwkv7')
    # sys.path.append(rwkv_path)
    rwkv6_path = os.path.join(parent_dir, 'rwkv')
    sys.path.append(rwkv6_path)
    rwkv_llama_path = os.path.join(parent_dir, 'rwkv_llama')
    sys.path.append(rwkv_llama_path)
    # print(f'add path: {rwkv_path} to sys.path')
    print(f'add path: {parent_dir} to sys.path')
    print(f'add path: {rwkv_llama_path} to sys.path')
    # os.environ['CUDA_HOME'] = '/usr/local/cuda-12.1'
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
    if 'RWKV_VERSION' not in os.environ:
        os.environ['RWKV_VERSION'] = 'v6'
setup_env()
from rwkv_llama.hybrid_vl_model import HybridModel,replace_llama_layers
from data.llava_data import LazySupervisedDataset, DataCollatorForSupervisedDataset,DataArguments
from llava.model.builder import load_pretrained_model
from torch.utils.data import DataLoader
import torch
from torch.utils.data.distributed import DistributedSampler
import yaml
import deepspeed
from train_functions import configure_optimizer, train_step, train_step_vl, validation_step
from profiler import timer, time_function
import time,math
import logging
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "DEBUG"),
    format='%(asctime)s.%(msecs)03d - PID:%(process)d : ThreadID:%(thread)d %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S %p'
)

logger = logging.getLogger(__name__)
#Create an argument parser to fill DataArguments
'''
@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)

    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)
    add_time_instruction: Optional[bool] = field(default=False)
    force_sample: Optional[bool] = field(default=False)
'''
import argparse
def create_args():
    parser = argparse.ArgumentParser()
    # Pretrained model path
    parser.add_argument('--pretrained_model', type=str, default='/home/yueyulin/models/lmms-lab/llava-onevision-qwen2-7b-si')
    parser.add_argument('--data_path', type=str, default='/home/yueyulin/data/MM_stage3/stage3_mini.json')
    parser.add_argument('--distill_config', type=str, default='configs/llava_qwen_7b.yaml')
    parser.add_argument('--output_dir', type=str, default='/tmp',help='directory to save the trained model')
    parser.add_argument('--max_seq_length', type=int, default=512, help='maximum sequence length to train the model')
    #Training arguments
    parser.add_argument('--micro_bsz', type=int, default=2)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--grad_cp', type=int, default=1)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    
    # 添加DeepSpeed相关的参数
    parser.add_argument('--deepspeed', action='store_true', help='Enable DeepSpeed')
    parser.add_argument('--deepspeed_config', type=str, default=None, help='Path to DeepSpeed config file')
    parser.add_argument('--deepspeed_stage', type=int, default=3, choices=[0, 1, 2, 3], help='DeepSpeed ZeRO stage')
    parser.add_argument('--deepspeed_offload', action='store_true', help='Enable CPU offloading')
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    
    #Optimizer arguments
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
    parser.add_argument('--log_every_n_steps', type=int, default=5000, help='number of steps after which the training progress will be logged')
    parser.add_argument('--is_sft', action='store_true', help='whether doing SFT',default=False)
    parser.add_argument('--wandb', type=str, default='hybrid_trainer_VL', help='wandb project name')
    parser.add_argument('--run_name', type=str, default='hybrid_trainer_VL_4090', help='run name for wandb logging')
    parser.add_argument('--save_per_batches', type=int, default=10000, help='number of batches to save the model')
    return parser

def load_configurations():
    args = create_args().parse_args()
    pretrained_model_path = args.pretrained_model
    config_file = os.path.join(pretrained_model_path, 'config.json')
    with open(config_file) as f:
        config = json.load(f)
    data_args = DataArguments(data_path=args.data_path)
    data_args.lazy_preprocess = True
    data_args.is_multimodal = True
    data_args.early_mix_text = False
    data_args.image_folder = ''
    data_args.image_aspect_ratio = config['image_aspect_ratio']
    data_args.image_grid_pinpoints = config['image_grid_pinpoints']
    data_args.image_crop_resolution = config['image_crop_resolution']
    data_args.image_split_resolution = config['image_split_resolution']
    data_args.mm_use_im_start_end = config['mm_use_im_start_end']
    
    args.betas = (args.beta1, args.beta2)
    return data_args,args
def lr_schedule(args, step):
    w_step = args.warmup_steps
    if args.lr_final == args.lr_init or args.epoch_count == 0:
        return args.lr_init
    
    decay_step = step
    decay_total = args.epoch_count * args.epoch_steps
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
    return args.weight_decay

def on_train_batch_start(args, model_engine, global_step, epoch):
    real_step = global_step + args.epoch_begin * args.epoch_steps

    # LR schedule
    lr = lr_schedule(args, real_step)
    
    # Weight decay schedule
    progress = (real_step - args.warmup_steps + 1) / (args.epoch_count  * args.epoch_steps - args.warmup_steps)
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
pbar = None
total_tokens = 0.0
last_total_tokens = 0.0
total_loss = 0
total_updates = 0
def on_train_batch_end(args, batch_idx, model_engine, loss, teacher_loss, kl_loss, student_cross_entropy_loss, global_step, epoch, last_log_time, token_per_step, is_accumulation_step, pbar):
    global total_tokens
    global last_total_tokens
    global total_loss
    global total_updates
    total_tokens += token_per_step
    current_time = time.time()
    # 只在实际更新参数时更新进度条
    if is_accumulation_step and model_engine.global_rank == 0:
        if pbar is None:
            pbar = tqdm(total=args.epoch_steps, desc=f"Epoch {epoch}")
        
        elapsed_time = current_time - last_log_time
        steps_per_second = args.accumulate_grad_batches / elapsed_time
        kt_s = (total_tokens - last_total_tokens) / elapsed_time / 1e3
        last_total_tokens = total_tokens
        pbar.update(1)
        total_loss += loss
        total_updates += 1
        pbar.set_postfix({
            'loss': f'{total_loss / total_updates:.4f}',
            'steps/s': f'{steps_per_second:.2f}',
            'kt/s': f'{kt_s:.2f}'
        })
        timer.print_stats(global_step)
        if args.wandb:
            wandb.log({
                "loss": loss,
                "lr": model_engine.optimizer.param_groups[0]['lr'],
                "weight_decay": model_engine.optimizer.param_groups[0]['weight_decay'],
                "steps_per_second": steps_per_second,
                "kt/s": kt_s,
                "global_step": global_step,
                "Gtokens": total_tokens / 1e9,
                "epoch": epoch,
                "teacher_loss": teacher_loss,
                "kl_loss": kl_loss,
                "student_cross_entropy_loss": student_cross_entropy_loss,
            })

    real_step = batch_idx
    if real_step % args.save_per_batches == 0 and real_step > 0 :
        #first check if the output_dir exists and deletes older checkpoints , we only keep latest 2 checkpoints
        if os.path.exists(args.output_dir):
            if model_engine.local_rank == 0:
                checkpoints = os.listdir(args.output_dir)
                #only list the directories   s
                checkpoints = [f for f in checkpoints if os.path.isdir(os.path.join(args.output_dir, f))]
                #sort by creation time  
                checkpoints.sort(key=lambda x: os.path.getctime(os.path.join(args.output_dir, x)))
                if len(checkpoints) > 2:
                    print(f'deleting older checkpoints {checkpoints[0]}')
                    import shutil
                    shutil.rmtree(os.path.join(args.output_dir, checkpoints[0]))    
        output_dir = f"{args.output_dir}/epoch_{epoch}_step_{real_step}"
        print(f'saving checkpoint to {output_dir}')
        try:
            model_engine.save_checkpoint(output_dir,f'epoch_{epoch}_step_{real_step}')
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
        import traceback
        traceback.print_exc()
        print(f'saved checkpoint to {output_dir}')

    return current_time, pbar
if __name__ == '__main__':
    data_args,args = load_configurations()
    print(data_args)
    model_name = "llava_qwen"
    llava_model_args = {
        "multimodal": True,
        "attn_implementation": "flash_attention_2",
    }
    with open(args.distill_config) as f:
        distill_config = yaml.load(f, Loader=yaml.FullLoader)
        print(distill_config)
      
    tokenizer, model, image_processor, max_length = load_pretrained_model(args.pretrained_model, None, model_name, device_map="cpu",torch_dtype="bfloat16", **llava_model_args)  # Add any other thing you want to pass in llava_model_args
    print(model)
    print(f'model device is {model.device},dtype is {model.dtype}')
    data_args.image_processor = image_processor
    #Configure the RWKV args
    # 设置参数
    args.my_pos_emb = 0
    args.head_size_a = 64
    args.head_size_divisor = 8
    args.ctx_len = 4096
    args.n_layer = model.config.num_hidden_layers
    args.n_embd = model.config.hidden_size
    args.dim_att = model.config.hidden_size
    args.dim_ffn = model.config.intermediate_size
    args.pre_ffn = 0
    args.head_qk = 0
    args.tiny_att_dim = 0
    args.tiny_att_layer = -999
    args.vocab_size = model.config.vocab_size
    args.layers = distill_config['RWKV']['layers']
    args.pad_id = tokenizer.pad_token_id
    args.kl_weight = distill_config['kl_weight']
    args.ce_weight = distill_config['ce_weight']
    args.is_llama_ffn = distill_config.get('is_llama_ffn', True)
    args.is_rwkv_att_only = distill_config.get('is_rwkv_att_only', False)
    args.is_all_labels_kl = distill_config.get('is_all_labels_kl', True)
    args.is_hidden_align = distill_config.get('is_hidden_align', False)
    args.dropout = 0.05
    print(args)
    model = replace_llama_layers(model,args)
    print(model)
    #update RWKVVLDecoderLayer only, freeze ffn
    for name,param in model.named_parameters():
        if 'block.' in name:
            if 'ffn' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        else:
            param.requires_grad = False
        print(f'{name}: {param.requires_grad}')
    _, teacher_model, _, _ = load_pretrained_model(args.pretrained_model, None, model_name, device_map="cpu", torch_dtype="bfloat16",**llava_model_args)  # Add any other thing you want to pass in llava_model_args
    print(f'teacher_model device is {teacher_model.device},dtype is {teacher_model.dtype}')
    teacher_model.eval()
    print(f'freeze teacher_model , teacher device is {teacher_model.device}')
    for name, param in teacher_model.named_parameters():
        param.requires_grad = False
    #Init deep speed engine for training
    ds = LazySupervisedDataset(args.data_path, tokenizer, data_args)
    collator = DataCollatorForSupervisedDataset(tokenizer)
    # dl = DataLoader(ds, batch_size=2, collate_fn=collator)
    train_sampler = DistributedSampler( 
            ds,
            num_replicas=args.world_size,
            rank=args.local_rank,
            shuffle=True
        )
    train_dataloader = torch.utils.data.DataLoader(
            ds, 
            batch_size=args.micro_bsz, 
            sampler=train_sampler,  # 使用分布式 sampler
            num_workers=4, 
            pin_memory=True, 
            drop_last=True, 
            collate_fn=collator
        )
    if args.deepspeed_config:
            # 如果提供了 DeepSpeed 配置文件，直接加载它
            with open(args.deepspeed_config, 'r') as f:
                ds_config = json.load(f)
    else:
        # 否则，根据命令行参数创建配置
        ds_config = {
            "train_batch_size": args.train_batch_size,
            "bf16": {
                "enabled": True
            },
            "fp32_reduce_scatter": True,
            "zero_optimization": {
                "stage": 3,
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_prefetch_bucket_size": 1e7,
                "stage3_param_persistence_threshold": 1e5,
                "memory_efficient_linear": True,
                "stage3_gather_16bit_weights_on_model_save": False,
                "zero_quantized_weights": False,
                "zero_hpz_partition_size": args.world_size,
                "zero_quantized_gradients": False,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True,
                    "buffer_count": 4,
                    "fast_init": True
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True,
                    "buffer_count": 5,
                    "buffer_size": 1e8,
                },
                "allgather_partitions": True,
                "sub_group_size": 1e9,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 1e7,
                "contiguous_gradients": True
            },
            "gradient_clipping": args.gradient_clip_val,
            "gradient_checkpointing": args.grad_cp == 1,
            "zero_force_ds_cpu_initialization": True,
            "zero_allow_untested_optimizer": True,
            "gradient_accumulation_steps": args.accumulate_grad_batches if args.accumulate_grad_batches > 1 else None,
            "wall_clock_breakdown": False
        }
    optimizer = configure_optimizer(model,args)
    print(optimizer)
    num_total_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'num_total_params: {num_total_params}, num_trainable_params: {num_trainable_params}, percent: {num_trainable_params / num_total_params}')
    #print current gpu memory
    print(f'current gpu memory BEFORE initializing deepspeed: {torch.cuda.memory_summary(device=None, abbreviated=False)}')
    print(f'initializing deepspeed with config {ds_config}')
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,  
            optimizer=optimizer,
            config=ds_config,
            model_parameters=model.parameters(),
    )
    timer.initialize_with_engine(model_engine)
    ds_config = {
                "train_batch_size": args.train_batch_size,
                "bf16": {
                    "enabled": True
                },
                "zero_optimization": {
                    "stage": 3,
                    "stage3_max_live_parameters": 1e9,
                    "stage3_max_reuse_distance": 1e9,
                    "stage3_prefetch_bucket_size": 5e6,
                    "memory_efficient_linear": True,
                    "stage3_param_persistence_threshold": 1e4,
                    "offload_param": {
                        "device": "cpu",
                        "pin_memory": True,
                        "buffer_count": 4,
                        "buffer_size": 1e8
                    },
                    "allgather_partitions": True,
                    "reduce_scatter": True,
                    "reduce_bucket_size": 5e6,
                    "overlap_comm": True,
                    "contiguous_gradients": True
                },
                "zero_force_ds_cpu_initialization": True
            }
    print(f'current gpu memory Before initializing teacher model: {torch.cuda.memory_summary(device=None, abbreviated=False)}')

    # 使用DeepSpeed包装teacher model
    teacher_engine, _, _, _ = deepspeed.initialize(
                model=teacher_model,
                config=ds_config,
                model_parameters=teacher_model.parameters(),
    )
    print(f'current gpu memory AFTER initializing teacher model: {torch.cuda.memory_summary(device=None, abbreviated=False)}')
    print(f'current gpu memory AFTER setting teacher model: {torch.cuda.memory_summary(device=None, abbreviated=False)}')
    del teacher_model
    torch.cuda.empty_cache()
    #print current gpu memory
    
    # 只在主进程上初始化wandb
    if args.wandb and model_engine.global_rank == 0:
        import wandb
        print(f'init wandb, project is {args.wandb}, name is {args.run_name}')
        wandb.init(project=args.wandb, name=args.run_name, config=args)
        print(f'begin training with {args.max_epochs} epochs')
        
    args.epoch_steps = len(train_dataloader) // (args.accumulate_grad_batches)
    global_step = 0
    last_log_time = time.time()
    tokenizer.model_max_length = args.max_seq_length#For DataCollatorForSupervisedDataset chunking
    # 训练循环
    for epoch in range(args.max_epochs):
        model_engine.train()
        if model_engine.global_rank == 0:
            from tqdm import tqdm
            pbar = tqdm(total=args.epoch_steps, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(train_dataloader):
            lr, wd_now = on_train_batch_start(args, model_engine, global_step, epoch)

            # batch = {k: v.to(model_engine.device) for k, v in batch.items()}
            images = batch['images']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            input_ids = input_ids.to(device=model_engine.device)
            attention_mask = attention_mask.to(device=model_engine.device)
            labels = labels.to(device=model_engine.device)
            images = [i.to(device=model_engine.device,dtype=torch.bfloat16) for i in images]
            batch['input_ids'] = input_ids
            batch['attention_mask'] = attention_mask
            batch['labels'] = labels
            batch['images'] = images
            print(f'input_ids: {input_ids.shape} image_sizes is {batch['image_sizes']} local_rank: {model_engine.local_rank}')
            
            
            # 前向传播
            loss, teacher_loss, kl_loss, student_cross_entropy_loss = train_step_vl(model_engine, batch, args, teacher_engine, tokenizer)
            
            token_per_step = input_ids.size(0) * input_ids.size(1)
            # 反向传���
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

        # 保存检查点
        if args.output_dir:
            model_engine.save_checkpoint(args.output_dir, f"checkpoint-epoch{epoch}")
            
        if pbar is not None:
            pbar.close()

    print("Training completed")
    if args.wandb and model_engine.global_rank == 0:
        wandb.finish()
    # model = model.to(device=device,dtype=torch.bfloat16)
    # for batch in dl:
    #     images = batch['images']
    #     input_ids = batch['input_ids']
    #     attention_mask = batch['attention_mask']
    #     labels = batch['labels']
    #     image_sizes = batch['image_sizes']
    #     input_ids = input_ids.to(device=device)
    #     attention_mask = attention_mask.to(device=device)
    #     labels = labels.to(device=device)
    #     images = [i.to(device=device) for i in images]
    #     with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
    #         with torch.no_grad():
    #             print('input_ids:',input_ids.shape)
    #             print('attention_mask:',attention_mask.shape)
    #             print('labels:',labels.shape)
    #             print('images:',images[0].shape)
    #             print('image_sizes:',image_sizes)
    #             output = model.forward(input_ids=input_ids,
    #                                 attention_mask=attention_mask,
    #                                 images=images,
    #                                 labels=labels,
    #                                 image_sizes=image_sizes,
    #                                 use_cache=False)
    #     print(output)
    # print(model.config)
    # logits = output.logits
    # loss = output.loss
    # print(logits.shape)
    # print(loss)