#!/usr/bin/env bash
set -euo pipefail
set -x

# ============================================================
# OpenOneRec Stage2 Full-Parameter Pretraining
# Single node, 8 GPUs
# torchrun + FSDP
# ============================================================

PRETRAIN_DIR=/home/jovyan/ceph-1/sujinsong/sujinsong/OpenOneRec-main/pretrain

# Stage1 转换后的 HuggingFace 模型
MODEL_DIR=${PRETRAIN_DIR}/model_output/stg1/step14000/global_step14000/converted

# Stage2 输出目录
OUTPUT_DIR=${PRETRAIN_DIR}/model_output/stg2

# 预训练数据配置
DATASET_CONFIG=examples/dataset_config/pretrain.json

cd "${PRETRAIN_DIR}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p /tmp/_wids_cache

# ============================================================
# 启动前检查
# ============================================================

if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "ERROR: Stage1 converted model directory does not exist:"
    echo "${MODEL_DIR}"
    exit 1
fi

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
    echo "ERROR: config.json does not exist:"
    echo "${MODEL_DIR}/config.json"
    exit 1
fi

if [[ ! -f "${MODEL_DIR}/tokenizer_config.json" ]]; then
    echo "ERROR: tokenizer_config.json does not exist:"
    echo "${MODEL_DIR}/tokenizer_config.json"
    exit 1
fi

if [[ ! -f "${DATASET_CONFIG}" ]]; then
    echo "ERROR: dataset config does not exist:"
    echo "${PRETRAIN_DIR}/${DATASET_CONFIG}"
    exit 1
fi

if [[ ! -f "${PRETRAIN_DIR}/torchrun_ompi_wrapper.py" ]]; then
    echo "ERROR: torchrun_ompi_wrapper.py does not exist:"
    echo "${PRETRAIN_DIR}/torchrun_ompi_wrapper.py"
    exit 1
fi

# ============================================================
# GPU 设置
# ============================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ============================================================
# 单机 NCCL 设置
# ============================================================

export NCCL_DEBUG=WARN

# 单机八卡训练关闭跨节点 IB
export NCCL_IB_DISABLE=1

# 保留 GPU P2P 和共享内存通信
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

# NCCL 异常处理
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ============================================================
# torchrun rendezvous
# ============================================================

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8499

# ============================================================
# 运行环境
# ============================================================

export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${PRETRAIN_DIR}:${PYTHONPATH:-}
export OMP_NUM_THREADS=1

# 注意：
# 不要 source set_env.sh
# set_env.sh 内部会调用 mpirun，当前环境没有 mpirun。

STDOUT_LOG=${OUTPUT_DIR}/stdout.log
STDERR_LOG=${OUTPUT_DIR}/stderr.log

SCRIPT_FILE=$(readlink -f "$0")

{
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "script: ${SCRIPT_FILE}"
    echo "stage: Stage2 full-parameter pretraining"
    echo "model_dir: ${MODEL_DIR}"
    echo "output_dir: ${OUTPUT_DIR}"
    echo "dataset_config: ${DATASET_CONFIG}"
    echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES}"
    echo "master_addr: ${MASTER_ADDR}"
    echo "master_port: ${MASTER_PORT}"
    echo "========================="
} >> "${OUTPUT_DIR}/task_info.log"

echo "============================================================"
echo "OpenOneRec Stage2 Full-Parameter Pretraining"
echo "============================================================"
echo "PRETRAIN_DIR=${PRETRAIN_DIR}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DATASET_CONFIG=${DATASET_CONFIG}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "STDOUT_LOG=${STDOUT_LOG}"
echo "STDERR_LOG=${STDERR_LOG}"
echo "============================================================"

# ============================================================
# Stage2 训练
#
# 说明：
# 1. MODEL_DIR 是 Stage1 converted HF 模型
# 2. 不使用 --freeze_llm，Stage2 全参数训练
# 3. 不使用 --resume_from
# 4. Stage2 从自己的 global_step=0 开始
# ============================================================

torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --max_restarts=0 \
    torchrun_ompi_wrapper.py recipes/train_qwen3.py \
        --model_dir "${MODEL_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --dataset_config "${DATASET_CONFIG}" \
        --use_tie_weights \
        --model_class Qwen3ForCausalLM \
        --monitor_datasource_loss \
        --monitor_datasource_cnt \
        --max_length 32768 \
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
        --use_chunked_loss_computer \
        > "${STDOUT_LOG}" \
        2> "${STDERR_LOG}"
