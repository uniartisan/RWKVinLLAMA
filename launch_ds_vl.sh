#!/bin/bash

# 设置环境变量
export MASTER_ADDR=192.168.1.35
export MASTER_PORT=29500
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eno1
export NCCL_DEBUG=INFO
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=NVL
export NCCL_SOCKET_FAMILY=AF_INET
export MAX_LEN=8192

# 获取当前节点的rank
NODE_RANK=$1

# 定义节点数和每个节点的GPU数量
NNODES=2
GPUS_PER_NODE_MASTER=4
GPUS_PER_NODE_WORKER=4
MICRO_BSZ=1
ACCUMULATE_GRAD_BATCHES=4
# 计算世界大小
WORLD_SIZE=$((GPUS_PER_NODE_MASTER + GPUS_PER_NODE_WORKER))
TRAIN_BATCH_SIZE=$((WORLD_SIZE * MICRO_BSZ * ACCUMULATE_GRAD_BATCHES))
# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# 根据节点rank选择正确的GPU数量
if [ $NODE_RANK -eq 0 ]; then
    GPUS_PER_NODE=$GPUS_PER_NODE_MASTER
else
    GPUS_PER_NODE=$GPUS_PER_NODE_WORKER
fi

torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_scripts/train_hybrid_vl_deepspeed.py \
    --deepspeed \
    --deepspeed_offload \
    --deepspeed_stage 3 \
    --config_file configs/test_hybrid_full_logits_qwenmlp.yaml \
    --output_dir /home/yueyulin/tmp/vl_2b \
    --data_path /home/yueyulin/data/MM_stage3/stage3.json \
    --num_devices $GPUS_PER_NODE \
    --num_nodes $NNODES \
    --micro_bsz $MICRO_BSZ \
    --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES \
    --max_epochs 2 \
    --wandb hybrid_trainer_vl_2B \
    --run_name "hybrid_trainer_vl_2B_${TIMESTAMP}" \
    --grad_cp 1 \
    --max_seq_length 8192 \
    --dropout 0.05 \
    --lr_init 1e-4 \
    --lr_final 1e-5 \
    --warmup_steps 1000 \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --save_per_batches 10000 \
    --world_size $WORLD_SIZE 
