#!/bin/bash

set -e

BASE_MODEL_DIR=$1
MODEL_HOME=$2
STEP=$3
CKPT_DIR=${MODEL_HOME}/step${STEP}/global_step${STEP}

OUTPUT_DIR=$CKPT_DIR/converted

python3 tools/model_converter/convert_checkpoint_to_hf.py --checkpoint_dir $CKPT_DIR \
    --output_dir $OUTPUT_DIR \
    --source_hf_model_path $BASE_MODEL_DIR
