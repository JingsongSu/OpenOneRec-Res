#!/usr/bin/env bash
set -euo pipefail
set -x

# OpenOneRec-Res Stage1 + Branch-Conditioned Interleaved Latent 3:
# train new itemic embedding rows, residual SID blocks, and three residual-like latent transition blocks for hard-branch thoughts before B/C/D.
PRETRAIN_DIR=/home/jovyan/ceph-1/sujinsong/sujinsong/OpenOneRec-latent/pretrain

# IMPORTANT:
# Branch-conditioned interleaved latent anchors do not need latent vocabulary expansion.
# Start from the clean itemic + feature base.
MODEL_DIR=${PRETRAIN_DIR}/model_output/Qwen3-0.6B_itemic_add_feature

OUTPUT_DIR=${PRETRAIN_DIR}/model_output/stg1_residual_add_feature_branch_interleaved_latent3
DATASET_CONFIG=${PRETRAIN_DIR}/examples/dataset_config/pretrain_residual_sid.json

ITEMIC_START_ID=151669

RESIDUAL_SID_NUM_LAYERS=4
RESIDUAL_SID_DROPOUT=0.1
RESIDUAL_SID_LOSS_WEIGHT=1.0

# 3 interleaved latent thoughts: after hard A/B/C, before formal B/C/D.
LATENT_REASONING_NUM_STEPS=3
LATENT_REASONING_DROPOUT=0.1
# No auxiliary latent CE. Formal residual B/C/D losses supervise each thought end-to-end.
LATENT_REASONING_LOSS_WEIGHT=0.0

MAX_LENGTH=${MAX_LENGTH:-13768}
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
  [[ -e "${required_path}" ]] || {
    echo "ERROR: missing ${required_path}" >&2
    exit 1
  }
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

# Base model only needs the original SID/itemic layout.
# There are no literal latent tokens in this experiment.
python tools/verify_residual_sid_layout.py \
  --model "${MODEL_DIR}" \
  --expected_layers "${RESIDUAL_SID_NUM_LAYERS}" \
  | tee "${SID_LAYOUT_JSON}"

TIE_WORD_EMBEDDINGS=$(python - "${MODEL_DIR}" <<'PY'
import sys
from transformers import AutoConfig

config = AutoConfig.from_pretrained(
    sys.argv[1],
    trust_remote_code=True,
)
print("true" if bool(config.tie_word_embeddings) else "false")
PY
)

USE_TIE_WEIGHTS_ARGS=()
[[ "${TIE_WORD_EMBEDDINGS}" != "true" ]] || \
  USE_TIE_WEIGHTS_ARGS+=(--use_tie_weights)

# The clean itemic base has neither residual SID blocks nor DPLR blocks.
# Generate the exact missing parameter names so Stage1 can initialize them.
#
# 4 SID layers -> 3 sid_residual_blocks
# Branch interleaved -> 3 latent_reasoning_blocks
ALLOW_RANDOM_INIT_PARAMS=$(
python - \
  "${RESIDUAL_SID_NUM_LAYERS}" \
  "${LATENT_REASONING_NUM_STEPS}" <<'PY'
import sys

num_residual_blocks = int(sys.argv[1]) - 1
num_latent_blocks = int(sys.argv[2])

suffixes = (
    "linear.weight",
    "linear.bias",
    "layer_norm.weight",
    "layer_norm.bias",
)

names = []

for i in range(num_residual_blocks):
    for suffix in suffixes:
        names.append(
            f"sid_residual_blocks.{i}.{suffix}"
        )

for i in range(num_latent_blocks):
    for suffix in suffixes:
        names.append(
            f"latent_reasoning_blocks.{i}.{suffix}"
        )

print(",".join(names))
PY
)

{
  echo "$(date '+%Y-%m-%d %H:%M:%S')"
  echo "script: ${SCRIPT_FILE}"
  echo "stage: residual_sid_stage1_branch_interleaved_latent3"
  echo "model_dir: ${MODEL_DIR}"
  echo "output_dir: ${OUTPUT_DIR}"
  echo "dataset_config: ${DATASET_CONFIG}"
  echo "max_length: ${MAX_LENGTH}"
  echo "tie_word_embeddings: ${TIE_WORD_EMBEDDINGS}"
  echo "sid_layout_json: ${SID_LAYOUT_JSON}"
  echo "latent_reasoning_num_steps: ${LATENT_REASONING_NUM_STEPS}"
  echo "latent_reasoning_dropout: ${LATENT_REASONING_DROPOUT}"
  echo "latent_reasoning_loss_weight: ${LATENT_REASONING_LOSS_WEIGHT}"
  echo "allow_random_init_params: ${ALLOW_RANDOM_INIT_PARAMS}"
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
  --allow_random_init_params "${ALLOW_RANDOM_INIT_PARAMS}" \
  --residual_sid_num_layers "${RESIDUAL_SID_NUM_LAYERS}" \
  --residual_sid_dropout "${RESIDUAL_SID_DROPOUT}" \
  --residual_sid_loss_weight "${RESIDUAL_SID_LOSS_WEIGHT}" \
  --mask_residual_sid_lm_loss \
  --latent_reasoning_num_steps "${LATENT_REASONING_NUM_STEPS}" \
  --latent_reasoning_dropout "${LATENT_REASONING_DROPOUT}" \
  --latent_reasoning_loss_weight "${LATENT_REASONING_LOSS_WEIGHT}" \
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
