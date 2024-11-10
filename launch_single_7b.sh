#!/bin/bash
NNODES=1
GPUS_PER_NODE=4
MICRO_BSZ=2
ACCUMULATE_GRAD_BATCHES=4
TRAIN_BATCH_SIZE=$((NNODES * GPUS_PER_NODE * MICRO_BSZ * ACCUMULATE_GRAD_BATCHES))
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
export RWKV_VERSION=v6
MAX_LENGTH=1024
deepspeed \
    --num_nodes $NNODES \
    --num_gpus $GPUS_PER_NODE \
    train_scripts/train_hybrid_deepspeed.py \
    --deepspeed \
    --deepspeed_offload \
    --deepspeed_stage 3 \
    --config_file configs/test_hybrid_full_logits_qwenmlp_remote.yaml \
    --output_dir /home/yueyulin/tmp/distill_qwen14b_1_distrib \
    --preprocessed_data /home/yueyulin/data/all_train_ds \
    --num_devices $GPUS_PER_NODE \
    --num_nodes $NNODES \
    --micro_bsz $MICRO_BSZ \
    --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES \
    --max_epochs 2 \
    --wandb hybrid_trainer \
    --run_name hybrid_trainer_7gpus_multinode \
    --grad_cp 1 \
    --max_seq_length $MAX_LENGTH \
    --dropout 0.05 \
    --lr_init 6e-4 \
    --lr_final 1e-5 \
    --warmup_steps 1000 \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --world_size $WORLD_SIZE