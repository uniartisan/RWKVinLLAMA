#!/bin/bash

CKPT_DIR=toys_playground/output
OUTPUT_DIR=toys_playground/output
while getopts "c:o:" opt; do
    case $opt in
        c) CKPT_DIR="$OPTARG";;
        o) OUTPUT_DIR="$OPTARG";;
    esac
done    
python $CKPT_DIR/zero_to_fp32.py -d --max_shard_size 10GB --exclude_frozen_parameters $CKPT_DIR $OUTPUT_DIR