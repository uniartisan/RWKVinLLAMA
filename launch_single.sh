#!/bin/bash
NNODES=1
GPUS_PER_NODE=4
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    train_scripts/train_hybrid_deepspeed.py \
    --deepspeed \
    --deepspeed_offload \
    --deepspeed_stage 3 \
    --config_file configs/test_hybrid_full_logits_qwenmlp.yaml \
    --output_dir /home/yueyulin/tmp/distill_qwen05b_1 \
    --preprocessed_data /home/yueyulin/data/all_train_ds \
    --num_devices $GPUS_PER_NODE \
    --num_nodes $NNODES \
    --micro_bsz 12 \
    --accumulate_grad_batches 4 \
    --max_epochs 2 \
    --wandb hybrid_trainer \
    --run_name hybrid_trainer_7gpus_multinode \
    --grad_cp 1 \
    --max_seq_length 512 \
    --dropout 0.05 \
    --lr_init 6e-4 \
    --lr_final 1e-5 \
    --warmup_steps 1000 \
    --train_batch_size 192 \
    --world_size 4