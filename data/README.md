# Dataset Documentation

This directory contains data processing scripts and dataset format specifications for the OpenOneRec project.

## Table of Contents

- [Quick Start](#quick-start) - Get started quickly with dataset download and processing
- [Directory Structure](#directory-structure)
- [Dataset Format Specification](#dataset-format-specification)
- [Notes](#notes)

## Directory Structure

- **general_text/**: General text data used in training, including pretraining and SFT datasets for mathematics, code, reasoning, and other domains
- **onerec_data/**: Recommendation scenario data and corresponding processing scripts that convert raw recommendation data into LLM pretraining and SFT training formats

### General Text Data (general_text)

The general text data directory contains information about the main general text datasets used in the project.

The `pretrain.csv` and `sft.csv` files list all HuggingFace dataset URLs and their corresponding sample counts. For easier reproduction, we have also released our processed datasets on HuggingFace:

- [Pretraining Dataset on HuggingFace](https://huggingface.co/datasets/OpenOneRec/OpenOneRec-General-Pretrain)
- [SFT Dataset on HuggingFace](https://huggingface.co/datasets/OpenOneRec/OpenOneRec-General-SFT)

> **NOTE**: The processed data on HuggingFace currently does not include some datasets (Nemotron_CC_Math_v1, Nemotron_Pretraining_Code_v1, Nemotron_CC_v2). We will provide a data processing script later to facilitate reproduction.

### OneRec Business Data (onerec_data)

The OneRec business data directory contains data processing scripts for recommendation systems, converting raw data into LLM pretraining and SFT training formats. It includes data processing scripts for various recommendation scenarios such as video recommendation, user profiling, interactive recommendation, label prediction, and cross-domain recommendation.

- [OpenOneRec Dataset on HuggingFace](https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF)

## Dataset Format Specification

To standardize data processing, we use a unified Parquet data format. Each Parquet file contains the following fields:

### Field Description

| Field | Type | Required | Default | Description | Requirements |
|-------|------|----------|---------|-------------|--------------|
| uuid | str | Yes | Auto-generated UUID | Unique identifier | Must be a valid UUID format, must be unique within the same dataset |
| source | str | Yes | - | Data source identifier | Cannot be an empty string |
| metadata | str | No | "{}" | JSON-formatted metadata dictionary | Must be a valid JSON dictionary string |
| images | str | No | "{}" | (Deprecated) This project only trains on text, this field is not used | - |
| videos | str | No | "{}" | (Deprecated) This project only trains on text, this field is not used | - |
| messages | str | No | None | JSON-formatted message list for conversation format data | Must be a valid JSON array, each message must have role and content fields |
| segments | str | No | None | JSON-formatted segment list for segmented data | Must be a valid JSON array, each segment must have a type field |
| image | str | No | None | (Deprecated) This project only trains on text, this field is not used | - |
| video | str | No | None | (Deprecated) This project only trains on text, this field is not used | - |
| text | str | No | None | Text content | No special requirements |
| label | str | No | None | Label information, if `image`, `video`, `text` exists, it is the corresponding label | No special requirements |

### Data Format Examples

The data format supports two main types:
- **Segments Format**: For regular text data, using the `segments` field to store text segment lists
- **Chat Format**: For conversation data, using the `messages` field to store conversation message lists

**Chat Format Data (Conversation Data):**

| Field | Value |
|-------|-------|
| uuid | 550e8400-e29b-41d4-a716-446655440001 |
| source | conversation_dataset |
| metadata | '{}' |
| images | '{}' |
| videos | '{}' |
| messages | '[{"role": "user", "content": [{"type": "text", "text": "What is machine learning?"}]}, {"role": "assistant", "content": [{"type": "text", "text": "Machine learning is a subset of artificial intelligence."}]}]' |

**Segments Format Data (Regular Text):**

| Field | Value |
|-------|-------|
| uuid | 550e8400-e29b-41d4-a716-446655440002 |
| source | document_dataset |
| metadata | '{}' |
| images | '{}' |
| videos | '{}' |
| segments | '[{"type": "text", "text": "Introduction paragraph..."}, {"type": "text", "text": "Main content..."}]' |

### Field Validation Rules

| Validation Item | Rule Description |
|-----------------|------------------|
| JSON Field Validation | metadata must be a valid JSON dictionary string; images and videos fields (deprecated) should be set to "{}" |
| Message Format Validation | messages field (if present) must contain a valid message list, each message must have role and content fields |
| Role Validation | Message role must be one of user, assistant, or system |
| Content Type Validation | The type in message content must be text (this project only trains on text, image and video types are not supported) |
| Segment Format Validation | segments field (if present) must contain a valid segment list, each segment must have a type field, type should be "text" |

### File Size Recommendations

For efficient DataLoader data loading, it is recommended that each Parquet file contains approximately **1000 samples**. If the data volume is large, you can use sharding to split the data into multiple files. The recommended file naming format is:

```
part-00000-of-00010.parquet
part-00001-of-00010.parquet
...
part-00009-of-00010.parquet
```

## Quick Start

### 1. Download Datasets

First, download the corresponding datasets from HuggingFace:

- [Pretraining General Text Dataset](https://huggingface.co/datasets/OpenOneRec/OpenOneRec-General-Pretrain)
- [SFT General Text Dataset](https://huggingface.co/datasets/OpenOneRec/OpenOneRec-General-SFT)
- [OneRec Recommendation Dataset](https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF)

You can download the datasets using the following commands (run from the **project root directory**):

```bash
pip install huggingface_hub

export HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>

hf download OpenOneRec/OpenOneRec-General-Pretrain \
    --repo-type dataset \
    --token $HF_TOKEN \
    --local-dir ./raw_data/general_text/pretrain

hf download OpenOneRec/OpenOneRec-General-SFT \
    --repo-type dataset \
    --token $HF_TOKEN \
    --local-dir ./raw_data/general_text/sft

hf download OpenOneRec/OpenOneRec-RecIF \
    --repo-type dataset \
    --token $HF_TOKEN \
    --local-dir ./raw_data/onerec_data
```

### 2. Process Recommendation Data

run:

```bash
cd data/onerec_data
bash run.sh
```

### 3. Pretraining Data Sharding

The generated data can be processed by calling the prepare scripts. Edit `prepare_pretrain.sh` or `prepare_sft.sh` and modify the following configuration:

```bash
GENERAL_TEXT_PATH="data/general_text"      # General text data path
REC_DATA_PATH="data/onerec_data/output"   # Recommendation data output path
OUTPUT_DIR="./output/split_data"          # Final output path
MAX_ROWS=1000                             # Number of samples per file
```

Then run:

```bash
# Process pretraining data
bash prepare_pretrain.sh

# Process SFT data
bash prepare_sft.sh
```

### 4. Distillation Data Processing

Data processing for on-policy distillation. Edit `prepare_distillation.sh` and modify the following configuration:

```bash
INPUT_PATH="data/general_text"                    # General text data path
OUTPUT_FILE="./output/onpolicy_distillation.parquet"  # Output file path
NUM_SAMPLES=200000                                # Number of samples to sample
SEED=42                                           # Random seed
```

Then run:

```bash
bash prepare_distillation.sh
```

### 5. RL Data Processing

Data processing for reinforcement learning (RL) training. Merges multiple RL task datasets and splits them into training and test sets. Edit `prepare_rl.sh` and modify the following configuration:

```bash
REC_DATA_PATH="data/onerec_data"                  # OneRec dataset path
OUTPUT_DIR="./output/rl_data"                     # Output directory path
TEST_SIZE=1000                                     # Number of test samples per subtask
SEED=42                                            # Random seed
```

The script processes the following 5 RL task datasets:
- `sft_video_rec.parquet` - Video recommendation task
- `sft_ad_rec.parquet` - Ad recommendation task
- `sft_product_rec.parquet` - Product recommendation task
- `sft_interactive_rec.parquet` - Interactive recommendation task
- `sft_label_cond_rec.parquet` - Label-conditioned recommendation task

Then run:

```bash
bash prepare_rl.sh
```

Output:
- `./output/rl_data/train.parquet` - Training set (remaining data after merging all tasks)
- `./output/rl_data/test.parquet` - Test set (1000 samples randomly sampled from merged data)


## Notes

* All scripts only process `split=0` (training set) data by default