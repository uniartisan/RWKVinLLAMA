#!/bin/bash

# 设置默认值
INPUT_DIR=toys_playground/raw_data
OUTPUT_DIR=toys_playground/dataset
TOKENIZER=/home/yueyulin/models/Qwen2.5-7B-Instruct/
MAX_LENGTH=2048
NUM_PROC=4  

# 解析命令行参数
while getopts "i:o:t:m:n:" opt; do
    case $opt in
        i) INPUT_DIR="$OPTARG";;
        o) OUTPUT_DIR="$OPTARG";;
        t) TOKENIZER="$OPTARG";;
        m) MAX_LENGTH="$OPTARG";;
        n) NUM_PROC="$OPTARG";;
        \?) echo "无效的选项 -$OPTARG" >&2; exit 1;;
    esac
done

mkdir -p $OUTPUT_DIR

for file in $INPUT_DIR/*; do
    echo $file
done
echo "start creating dataset"
echo "input_dir: $INPUT_DIR"
echo "output_dir: $OUTPUT_DIR"
echo "tokenizer: $TOKENIZER"
echo "max_len: $MAX_LENGTH"
echo "num_processes: $NUM_PROC"

python data/create_hybrid_dataset.py --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR --tokenizer $TOKENIZER --max_len $MAX_LENGTH --num_processes $NUM_PROC

python data/make_statistics.py --data_dir $OUTPUT_DIR