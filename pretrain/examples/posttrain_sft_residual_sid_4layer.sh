#!/usr/bin/env bash
set -euo pipefail
set -x

PRETRAIN_DIR=/home/jovyan/ceph-1/sujinsong/sujinsong/OpenOneRec-res/pretrain

# Stage2 转换后的 HuggingFace 模型
STAGE2_OUTPUT_DIR=${PRETRAIN_DIR}/model_output/stg2
MODEL_DIR=${STAGE2_OUTPUT_DIR}/step11500/global_step11500/converted

# Residual SFT 输出目录
# 建议不要与原始 SFT 输出目录共用，方便做基线对照。
OUTPUT_DIR=${PRETRAIN_DIR}/model_output/sft_residual_sid_4layer-1loss-mask

# SFT 数据配置
DATASET_CONFIG=${PRETRAIN_DIR}/examples/dataset_config/sft.json

# 四层 SID。层名称和每层大小由 tokenizer 自动发现。
RESIDUAL_SID_NUM_LAYERS=4
RESIDUAL_SID_DROPOUT=0.1
RESIDUAL_SID_LOSS_WEIGHT=1.0

cd "${PRETRAIN_DIR}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p /tmp/_wids_cache

if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "ERROR: Stage2 converted model directory does not exist:"
    echo "${MODEL_DIR}"
    exit 1
fi

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
    echo "ERROR: config.json not found:"
    echo "${MODEL_DIR}/config.json"
    exit 1
fi

if [[ ! -f "${DATASET_CONFIG}" ]]; then
    echo "ERROR: SFT dataset config not found:"
    echo "${DATASET_CONFIG}"
    exit 1
fi

if [[ ! -f "${PRETRAIN_DIR}/torchrun_ompi_wrapper.py" ]]; then
    echo "ERROR: torchrun_ompi_wrapper.py not found:"
    echo "${PRETRAIN_DIR}/torchrun_ompi_wrapper.py"
    exit 1
fi

if [[ ! -f "${PRETRAIN_DIR}/recipes/train_qwen3_residual_sid.py" ]]; then
    echo "ERROR: residual SID training recipe not found:"
    echo "${PRETRAIN_DIR}/recipes/train_qwen3_residual_sid.py"
    echo
    echo "Please install the four-layer residual SID autodiscovery patch first."
    exit 1
fi

if [[ ! -f "${PRETRAIN_DIR}/tools/verify_residual_sid_layout.py" ]]; then
    echo "ERROR: SID layout verifier not found:"
    echo "${PRETRAIN_DIR}/tools/verify_residual_sid_layout.py"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export NCCL_DEBUG=WARN

# 单机训练不需要跨节点 IB
export NCCL_IB_DISABLE=1

# 保留 GPU P2P 和共享内存通信
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

# NCCL 异常检测
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8499


export PYTHONPATH=${PRETRAIN_DIR}:${PYTHONPATH:-}
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1


STDOUT_LOG=${OUTPUT_DIR}/stdout.log
STDERR_LOG=${OUTPUT_DIR}/stderr.log

SCRIPT_FILE=$(readlink -f "$0")


SID_LAYOUT_JSON=${OUTPUT_DIR}/sid_layout.json

python tools/verify_residual_sid_layout.py \
    --model "${MODEL_DIR}" \
    --expected_layers "${RESIDUAL_SID_NUM_LAYERS}" \
    | tee "${SID_LAYOUT_JSON}"


TIE_WORD_EMBEDDINGS=$(
python - "${MODEL_DIR}" <<'PY'
import sys
from transformers import AutoConfig

model_dir = sys.argv[1]
config = AutoConfig.from_pretrained(
    model_dir,
    trust_remote_code=True,
)
print("true" if bool(config.tie_word_embeddings) else "false")
PY
)

USE_TIE_WEIGHTS_ARGS=()
if [[ "${TIE_WORD_EMBEDDINGS}" == "true" ]]; then
    USE_TIE_WEIGHTS_ARGS+=(--use_tie_weights)
fi


{
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "script: ${SCRIPT_FILE}"
    echo "stage: residual_sid_sft"
    echo "model_dir: ${MODEL_DIR}"
    echo "output_dir: ${OUTPUT_DIR}"
    echo "dataset_config: ${DATASET_CONFIG}"
    echo "residual_sid_num_layers: ${RESIDUAL_SID_NUM_LAYERS}"
    echo "residual_sid_dropout: ${RESIDUAL_SID_DROPOUT}"
    echo "residual_sid_loss_weight: ${RESIDUAL_SID_LOSS_WEIGHT}"
    echo "tie_word_embeddings: ${TIE_WORD_EMBEDDINGS}"
    echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES}"
    echo "master_addr: ${MASTER_ADDR}"
    echo "master_port: ${MASTER_PORT}"
    echo "sid_layout_json: ${SID_LAYOUT_JSON}"
    echo "========================="
} >> "${OUTPUT_DIR}/task_info.log"

echo "============================================================"
echo "OpenOneRec Four-Layer Residual SID SFT"
echo "============================================================"
echo "PRETRAIN_DIR=${PRETRAIN_DIR}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DATASET_CONFIG=${DATASET_CONFIG}"
echo "RESIDUAL_SID_NUM_LAYERS=${RESIDUAL_SID_NUM_LAYERS}"
echo "RESIDUAL_SID_DROPOUT=${RESIDUAL_SID_DROPOUT}"
echo "RESIDUAL_SID_LOSS_WEIGHT=${RESIDUAL_SID_LOSS_WEIGHT}"
echo "TIE_WORD_EMBEDDINGS=${TIE_WORD_EMBEDDINGS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "SID_LAYOUT_JSON=${SID_LAYOUT_JSON}"
echo "STDOUT_LOG=${STDOUT_LOG}"
echo "STDERR_LOG=${STDERR_LOG}"
echo "============================================================"


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
        "${USE_TIE_WEIGHTS_ARGS[@]}" \
        --model_class Qwen3ForCausalLMResidualSID \
        --allow_random_init_params sid_residual_blocks \
        --residual_sid_num_layers "${RESIDUAL_SID_NUM_LAYERS}" \
        --residual_sid_dropout "${RESIDUAL_SID_DROPOUT}" \
        --residual_sid_loss_weight "${RESIDUAL_SID_LOSS_WEIGHT}" \
        --mask_residual_sid_lm_loss \
        --monitor_datasource_loss \
        --monitor_datasource_cnt \
        --max_length 28768 \
        --learning_rate 2e-4 \
        --min_lr 1e-4 \
        --weight_decay 0.1 \
        --max_grad_norm 1.0 \
        --lr_scheduler_type cosine \
        --num_warmup_steps 500 \
        --num_training_steps 5000 \
        --save_checkpoint_per_step 500 \
        --minibatch_size 12384 \
        --logging_per_step 50 \
        --use_fp32_weight \
        --seed 19260817 \
        --enable_profiler \
        --enable_gradient_checkpointing \
        > "${STDOUT_LOG}" \
        2> "${STDERR_LOG}"

