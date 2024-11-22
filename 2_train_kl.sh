#!/bin/bash
NNODES=1
GPUS_PER_NODE=4
MICRO_BSZ=1
ACCUMULATE_GRAD_BATCHES=4
export RWKV_VERSION=v6
MAX_LENGTH=512
CONFIG_FILE=toys_playground/configs/qwen7B_KL_Local.yaml
OUTPUT_DIR=toys_playground/output
PREPROCESSED_DATA=toys_playground/dataset
LR_INIT=6e-4
LR_FINAL=1e-5
WARMUP_STEPS=1000

while getopts "c:o:p:n:m:b:a:l:f:w:" opt; do
    case $opt in
        c) CONFIG_FILE="$OPTARG";;
        o) OUTPUT_DIR="$OPTARG";;
        p) PREPROCESSED_DATA="$OPTARG";;
        n) NNODES="$OPTARG";;
        m) MAX_LENGTH="$OPTARG";;
        b) MICRO_BSZ="$OPTARG";;
        a) ACCUMULATE_GRAD_BATCHES="$OPTARG";;
        l) LR_INIT="$OPTARG";;
        f) LR_FINAL="$OPTARG";;
        w) WARMUP_STEPS="$OPTARG";;
        \?) echo "无效的选项 -$OPTARG" >&2; exit 1;;
    esac
done

TRAIN_BATCH_SIZE=$((NNODES * GPUS_PER_NODE * MICRO_BSZ * ACCUMULATE_GRAD_BATCHES))
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
deepspeed \
    --num_nodes $NNODES \
    --num_gpus $GPUS_PER_NODE \
    train_scripts/train_hybrid_deepspeed.py \
    --deepspeed \
    --deepspeed_offload \
    --deepspeed_stage 3 \
    --config_file $CONFIG_FILE \
    --output_dir $OUTPUT_DIR \
    --preprocessed_data $PREPROCESSED_DATA \
    --num_devices $GPUS_PER_NODE \
    --num_nodes $NNODES \
    --micro_bsz $MICRO_BSZ \
    --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES \
    --max_epochs 2 \
    --wandb hybrid_trainer_toys \
    --run_name hybrid_trainer_toys \
    --grad_cp 1 \
    --max_seq_length $MAX_LENGTH \
    --dropout 0.05 \
    --lr_init $LR_INIT \
    --lr_final $LR_FINAL \
    --warmup_steps $WARMUP_STEPS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --world_size $WORLD_SIZE