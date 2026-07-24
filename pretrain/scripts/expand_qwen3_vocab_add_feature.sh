#!/bin/bash

set -e

HF_MODEL_DIR=/home/jovyan/ceph-1/sujinsong/sujinsong/OpenOneRec-res/pretrain/model_output/Qwen3-0.6B
OUTPUT_MODEL_DIR=/home/jovyan/ceph-1/sujinsong/sujinsong/OpenOneRec-res/pretrain/model_output/Qwen3-0.6B_itemic_add_feature
ITEMIC_LAYER_N=4
VOCAB_SIZE_PER_LAYER=1024
LAST_LAYER_VOCAB_SIZE=15000

CTYPE_TOKEN_SIZE=18
MID_VOCAB_SIZE_PER_LAYER=1024

# 不添加任何 LS token
LS_VOCAB_SIZE_PER_LAYER=0

TIME_TOKEN_SIZE=0

python3 tools/model_converter/expand_qwen3_vocab_add_feature.py \
  --hf_model_dir "${HF_MODEL_DIR}" \
  --output_model_dir "${OUTPUT_MODEL_DIR}" \
  --itemic_layer_n "${ITEMIC_LAYER_N}" \
  --vocab_size_per_layer "${VOCAB_SIZE_PER_LAYER}" \
  --last_layer_vocab_size "${LAST_LAYER_VOCAB_SIZE}" \
  --time_token_size "${TIME_TOKEN_SIZE}" \
  --ctype_token_size "${CTYPE_TOKEN_SIZE}" \
  --mid_vocab_size_per_layer "${MID_VOCAB_SIZE_PER_LAYER}" \
  --ls_vocab_size_per_layer "${LS_VOCAB_SIZE_PER_LAYER}"