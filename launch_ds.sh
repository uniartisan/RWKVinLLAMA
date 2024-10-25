#!/bin/bash

# 设置环境变量
export MASTER_ADDR=192.168.1.35
export MASTER_PORT=29500
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=ibp2s0f0,ibp2s0f1
export NCCL_DEBUG=INFO
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=NVL

# 获取当前节点的rank
NODE_RANK=$1

# 定义节点数和每个节点的GPU数量
NNODES=2
GPUS_PER_NODE_MASTER=4
GPUS_PER_NODE_WORKER=3

# 计算世界大小
WORLD_SIZE=$((GPUS_PER_NODE_MASTER + GPUS_PER_NODE_WORKER))

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
    train_scripts/train_hybrid_deepspeed.py \
    --deepspeed \
    --deepspeed_offload \
    --deepspeed_stage 3 \
    --config_file configs/step_wise/test_hybrid_1_layer_qwenmlp-0.5B.yaml \
    --output_dir /home/yueyulin/tmp/distill_qwen05b_1 \
    --preprocessed_data /home/yueyulin/data/IndustryInstruction_Qwen_0.5B_DS/ \
    --num_devices $GPUS_PER_NODE \
    --num_nodes $NNODES \
    --micro_bsz 1 \
    --accumulate_grad_batches 4 \
    --max_epochs 2 \
    --wandb hybrid_trainer \
    --run_name hybrid_trainer_7gpus_multinode \
    --grad_cp 1 \
    --max_seq_length 2048 \
    --dropout 0.05 \
    --lr_init 6e-4 \
    --lr_final 1e-5 \
    --warmup_steps 1000 \
    --train_batch_size 28