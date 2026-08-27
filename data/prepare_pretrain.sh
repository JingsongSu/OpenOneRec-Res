#!/bin/bash
# Data splitting script: Merge general text and recommendation data, then split by every 1000 samples

set -e

# Configuration
# Both general and onerec use datasets starting with pretrain
GENERAL_TEXT_PATH="/home/jovyan/ceph-1/sujinsong/sujinsong/OpenOneRec-res/raw_data/general_text/pretrain/empty"
# GENERAL_TEXT_PATH="../raw_data/general_text/pretrain/OpenOneRec-General-Pretrain"
REC_DATA_PATH="/home/jovyan/ceph-1/sujinsong/sujinsong/OpenOneRec-res/output/only_seq_add_feature_time/pretrain"
OUTPUT_DIR="/home/jovyan/ceph-1/sujinsong/sujinsong/OpenOneRec-res/output/split_data_pretrain_only_seq_add_feature_time"
MAX_ROWS=100000
ENGINE="pyarrow"

# Check if paths exist
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -e "${GENERAL_TEXT_PATH}" ]; then
    echo "Error: General text path does not exist: ${GENERAL_TEXT_PATH}"
    exit 1
fi

if [ ! -e "${REC_DATA_PATH}" ]; then
    echo "Error: Recommendation data path does not exist: ${REC_DATA_PATH}"
    exit 1
fi

# Execute
python3 "${SCRIPT_DIR}/scripts/split_data.py" \
    --general_text_path "${GENERAL_TEXT_PATH}" \
    --rec_data_path "${REC_DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_rows "${MAX_ROWS}" \
    --engine "${ENGINE}"

