# OneRec Data Processing Scripts

This directory contains data processing scripts for the OneRec project, converting raw data into LLM pretraining and SFT training formats.

## Directory Structure

```
data/
├── pretrain/               # Pretrain data processing scripts
│   ├── video_rec.py        # Video recommendation pretrain
│   ├── user_profile.py     # User profile pretrain
│   └── item_understand.py  # item understanding alignment pretrain
├── sft/                    # SFT data processing scripts
│   ├── video_rec.py        # Video recommendation
│   ├── interactive_rec.py  # Interactive recommendation
│   ├── label_cond_rec.py   # Label conditional recommendation
│   ├── label_pred.py       # Label prediction (binary classification)
│   ├── ad_rec.py           # Ad recommendation (cross-domain)
│   ├── product_rec.py      # Product recommendation (cross-domain)
│   ├── item_understand.py      # Item understand 
│   └── reco_reason.py      # Recommendation reasoning
├── run.sh                  # Main execution script
└── README.md
```

## Quick Start

### 1. Configure Input Paths

Edit `run.sh` to set the following paths:

```bash
INPUT_METADATA="path/to/onerec_bench_release.parquet"
PID2SID_MAPPING="path/to/video_ad_pid2sid.parquet"
PRODUCT_PID2SID_MAPPING="path/to/product_pid2sid.parquet"
CAPTION_INPUT="path/to/pid2caption.parquet"
OUTPUT_BASE_DIR="./output"
```

### 2. Select Tasks to Run

Uncomment the tasks you want to run in `run.sh`:

```bash
# Pretrain tasks
RUN_PRETRAIN_VIDEO_REC=1
RUN_PRETRAIN_USER_PROFILE=1
RUN_PRETRAIN_SID2CAPTION=1

# SFT tasks
RUN_SFT_VIDEO_REC=1
RUN_SFT_INTERACTIVE_REC=1
# ...
```

### 3. Run

```bash
cd data
bash run.sh
```

## Task Descriptions

### Pretrain Tasks

| Task | Script | Description |
|------|--------|-------------|
| video_rec | `pretrain/video_rec.py` | Concatenate user history SID sequence with target SID sequence for sequence modeling pretrain |
| user_profile | `pretrain/user_profile.py` | Use `inter_user_profile_with_sid` field as pretrain text |
| item_understand | `pretrain/item_understand.py` | Build item understanding alignment data using various template formats |

### SFT Tasks

| Task | Script | Description |
|------|--------|-------------|
| video_rec | `sft/video_rec.py` | Predict next video based on user browsing history |
| interactive_rec | `sft/interactive_rec.py` | Recommend content based on user profile and search keywords |
| label_cond_rec | `sft/label_cond_rec.py` | Predict items by interaction type (like/follow/forward/etc.) |
| label_pred | `sft/label_pred.py` | Binary classification: predict if user will watch a video for long |
| ad_rec | `sft/ad_rec.py` | Cross-domain: predict ad clicks based on video and ad history |
| product_rec | `sft/product_rec.py` | Cross-domain: predict product clicks based on video and product history |
| item_understand | `sft/item_understand.py` | Generate video description from SID |
| reco_reason | `sft/reco_reason.py` | Generate recommendation reasoning: analyze user interests and explain recommendations |

## Output Format

### Pretrain Format

```json
{
  "source": "RecIF_VideoRec_Pretrain",
  "uuid": "xxx",
  "segments": [{"type": "text", "text": "..."}],
  "metadata": {"uid": 123}
}
```

### SFT Format

```json
{
  "source": "RecIF_VideoRec",
  "uuid": "xxx",
  "messages": [
    {"role": "system", "content": [{"type": "text", "text": "..."}]},
    {"role": "user", "content": [{"type": "text", "text": "..."}]},
    {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
  ],
  "metadata": {"uid": 123}
}
```

## SID Format

All scripts use a unified SID format:

```
<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>
```

Where `c0`, `c1`, `c2` are triplet codes obtained from the `pid2sid` mapping table.

## Dependencies

- pandas
- numpy
- tqdm

## Running Individual Scripts

Each script can also be run independently:

```bash
# Example: Run video_rec SFT task
python sft/video_rec.py \
    --input /path/to/metadata.parquet \
    --pid2sid /path/to/pid2sid.parquet \
    --output_dir ./output \
    --seed 42
```

## Notes

1. All scripts process only `split=0` (training set) data by default
2. Output files are named as `{task_type}_{task_name}.parquet`
3. Cross-domain tasks (product_rec) require additional pid2sid mapping files
