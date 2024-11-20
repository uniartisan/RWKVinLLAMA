#!/bin/bash
MODEL_PATH="/home/yueyulin/models/Qwen2.5-7B-Instruct/"
WORLD_SIZE=5
NUM_GPUS=3
PAD_ID=151645
BATCH_SIZE=2
MAX_LENGTH=1024
NUM_LAYERS=28
OUTPUT_HIDDEN=
DEVICE_MAP="server/device_map.json"
while getopts ":m:w:g:p:b:l:o:h:d:" opt; do
    case $opt in
        m) MODEL_PATH="$OPTARG"
        ;;
        w) WORLD_SIZE="$OPTARG"
        ;;
        g) NUM_GPUS="$OPTARG"
        ;;
        p) PAD_ID="$OPTARG"
        ;;
        b) BATCH_SIZE="$OPTARG"
        ;;
        l) MAX_LENGTH="$OPTARG"
        ;;
        o) NUM_LAYERS="$OPTARG"
        ;;
        h) OUTPUT_HIDDEN="--output_all_hiddens"
        ;;
        d) DEVICE_MAP="$OPTARG"
        ;;
    esac
done

echo "program parameters:"
echo "MODEL_PATH: $MODEL_PATH"
echo "NUM_LAYERS: $NUM_LAYERS"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "NUM_GPUS: $NUM_GPUS"
echo "PAD_ID: $PAD_ID"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "MAX_LENGTH: $MAX_LENGTH"
echo "DEVICE_MAP: $DEVICE_MAP"

echo "Starting teacher server..."

python server/teacher_server_nccl_distill.py --model_id $MODEL_PATH --batch $BATCH_SIZE --length $MAX_LENGTH --size $WORLD_SIZE  --num_gpus $NUM_GPUS --num_layers $NUM_LAYERS $OUTPUT_HIDDEN --device_map $DEVICE_MAP  
 

