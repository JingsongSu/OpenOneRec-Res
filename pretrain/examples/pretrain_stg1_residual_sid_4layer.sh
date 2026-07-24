#!/usr/bin/env bash
set -euo pipefail
set -x

# OpenOneRec-Res Stage1: train new itemic embedding rows and residual SID blocks.
PRETRAIN_DIR=/home/jovyan/ceph-1/sujinsong/sujinsong/OpenOneRec-res/pretrain
MODEL_DIR=${PRETRAIN_DIR}/model_output/Qwen3-0.6B_itemic_add_feature
OUTPUT_DIR=${PRETRAIN_DIR}/model_output/stg1_residual_add_feature
DATASET_CONFIG=${PRETRAIN_DIR}/examples/dataset_config/pretrain_residual_sid.json
ITEMIC_START_ID=151669
RESIDUAL_SID_NUM_LAYERS=4
RESIDUAL_SID_DROPOUT=0.1
RESIDUAL_SID_LOSS_WEIGHT=1.0
MAX_LENGTH=${MAX_LENGTH:-22768}
MASTER_PORT=${MASTER_PORT:-8499}

cd "${PRETRAIN_DIR}"
mkdir -p "${OUTPUT_DIR}" /tmp/_wids_cache
for required_path in \
  "${MODEL_DIR}/config.json" \
  "${MODEL_DIR}/tokenizer_config.json" \
  "${DATASET_CONFIG}" \
  "${PRETRAIN_DIR}/torchrun_ompi_wrapper.py" \
  "${PRETRAIN_DIR}/recipes/train_qwen3_residual_sid.py" \
  "${PRETRAIN_DIR}/tools/verify_residual_sid_layout.py"
do
  [[ -e "${required_path}" ]] || { echo "ERROR: missing ${required_path}" >&2; exit 1; }
done

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${PRETRAIN_DIR}:${PYTHONPATH:-}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

STDOUT_LOG=${OUTPUT_DIR}/stdout.log
STDERR_LOG=${OUTPUT_DIR}/stderr.log
SID_LAYOUT_JSON=${OUTPUT_DIR}/sid_layout.json
SCRIPT_FILE=$(readlink -f "$0")

python tools/verify_residual_sid_layout.py \
  --model "${MODEL_DIR}" \
  --expected_layers "${RESIDUAL_SID_NUM_LAYERS}" \
  | tee "${SID_LAYOUT_JSON}"

TIE_WORD_EMBEDDINGS=$(python - "${MODEL_DIR}" <<'PY'
import sys
from transformers import AutoConfig
config = AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True)
print("true" if bool(config.tie_word_embeddings) else "false")
PY
)
USE_TIE_WEIGHTS_ARGS=()
[[ "${TIE_WORD_EMBEDDINGS}" != "true" ]] || USE_TIE_WEIGHTS_ARGS+=(--use_tie_weights)

{
  echo "$(date '+%Y-%m-%d %H:%M:%S')"
  echo "script: ${SCRIPT_FILE}"
  echo "stage: residual_sid_stage1"
  echo "model_dir: ${MODEL_DIR}"
  echo "output_dir: ${OUTPUT_DIR}"
  echo "dataset_config: ${DATASET_CONFIG}"
  echo "max_length: ${MAX_LENGTH}"
  echo "tie_word_embeddings: ${TIE_WORD_EMBEDDINGS}"
  echo "sid_layout_json: ${SID_LAYOUT_JSON}"
  echo "========================="
} >> "${OUTPUT_DIR}/task_info.log"

torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  --max_restarts=0 \
  torchrun_ompi_wrapper.py recipes/train_qwen3_residual_sid.py \
  --model_dir "${MODEL_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --dataset_config "${DATASET_CONFIG}" \
  --freeze_llm \
  "${USE_TIE_WEIGHTS_ARGS[@]}" \
  --start_optimize_embedding_index "${ITEMIC_START_ID}" \
  --model_class Qwen3ForCausalLMResidualSID \
  --allow_random_init_params sid_residual_blocks \
  --residual_sid_num_layers "${RESIDUAL_SID_NUM_LAYERS}" \
  --residual_sid_dropout "${RESIDUAL_SID_DROPOUT}" \
  --residual_sid_loss_weight "${RESIDUAL_SID_LOSS_WEIGHT}" \
  --mask_residual_sid_lm_loss \
  --monitor_datasource_loss \
  --monitor_datasource_cnt \
  --max_length "${MAX_LENGTH}" \
  --learning_rate 2e-4 \
  --min_lr 1e-4 \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --lr_scheduler_type cosine \
  --num_warmup_steps 200 \
  --num_training_steps 2000 \
  --save_checkpoint_per_step 500 \
  --minibatch_size 12384 \
  --logging_per_step 50 \
  --use_fp32_weight \
  --seed 19260817 \
  --enable_profiler \
  --enable_gradient_checkpointing \
  > "${STDOUT_LOG}" \
  2> "${STDERR_LOG}"
