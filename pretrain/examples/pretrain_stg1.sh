#!/usr/bin/env bash
set -e
set -x

# ============================================================
# OpenOneRec Stage1 Pretrain
# Single machine, 8 GPUs, torchrun launcher, no mpirun
# ============================================================

PRETRAIN_DIR=/home/jovyan/ceph-1/sujinsong/sujinsong/OpenOneRec-main/pretrain

MODEL_DIR=${PRETRAIN_DIR}/model_output/Qwen3-0.6B_itemic
OUTPUT_DIR=${PRETRAIN_DIR}/model_output/stg1
DATASET_CONFIG=examples/dataset_config/pretrain.json

cd ${PRETRAIN_DIR}

mkdir -p ${OUTPUT_DIR}
mkdir -p /tmp/_wids_cache

# 使用 0-7 八张卡，对应原来的 mpirun -np 8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 单机多卡 NCCL 设置
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

# torchrun 单机 rendezvous 设置
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8499

# 其他环境变量
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${PRETRAIN_DIR}:${PYTHONPATH:-}

STDOUT_LOG=${OUTPUT_DIR}/stdout.log
STDERR_LOG=${OUTPUT_DIR}/stderr.log

SCRIPT_FILE=$(readlink -f "$0")

echo "$(date '+%Y-%m-%d %H:%M:%S')" >> ${OUTPUT_DIR}/task_info.log
echo "script: ${SCRIPT_FILE}" >> ${OUTPUT_DIR}/task_info.log
echo "=========================" >> ${OUTPUT_DIR}/task_info.log

echo "============================================================"
echo "Stage1 Pretrain"
echo "PRETRAIN_DIR=${PRETRAIN_DIR}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DATASET_CONFIG=${DATASET_CONFIG}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "============================================================"

torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  torchrun_ompi_wrapper.py recipes/train_qwen3.py \
    --model_dir ${MODEL_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --dataset_config ${DATASET_CONFIG} \
    --freeze_llm \
    --use_tie_weights \
    --start_optimize_embedding_index 151669 \
    --model_class Qwen3ForCausalLM \
    --monitor_datasource_loss \
    --monitor_datasource_cnt \
    --max_length 32768 \
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
    --use_chunked_loss_computer \
    > ${STDOUT_LOG} 2> ${STDERR_LOG}
    