#!/bin/bash
# Data splitting script: Merge general text and recommendation data, then split by every 1000 samples

set -e

# Configuration
# Both general and onerec use datasets starting with sft
GENERAL_TEXT_PATH="/home/jovyan/ceph-1/sujinsong/online/openonerec-res-v2/raw_data/general_text/sft/empty"
# GENERAL_TEXT_PATH="../raw_data/general_text/sft/OpenOneRec-General-SFT"
REC_DATA_PATH="/home/jovyan/ceph-1/sujinsong/online/openonerec-res-v2/output/only_seq_add_feature_time/sft"   ## 这里要改
OUTPUT_DIR="/home/jovyan/ceph-1/sujinsong/online/openonerec-res-v2/output/split_data_sft_only_seq_add_feature_time_daily"
MAX_ROWS=100000
ENGINE="pyarrow"

#删除旧的日更数据
rm -rf $OUTPUT_DIR

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

