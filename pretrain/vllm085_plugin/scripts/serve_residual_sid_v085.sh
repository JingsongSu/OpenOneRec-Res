#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/path/to/sft_residual_4layer_vllm085_hf}"
SERVED_MODEL="${SERVED_MODEL:-onerec-residual-sid}"
TP="${TP:-1}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

export VLLM_USE_V1=0
export VLLM_PLUGINS=openonerec_residual_sid_v085

vllm serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL}" \
  --task embed \
  --tensor-parallel-size "${TP}" \
  --pipeline-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --trust-remote-code \
  --port "${PORT}"
