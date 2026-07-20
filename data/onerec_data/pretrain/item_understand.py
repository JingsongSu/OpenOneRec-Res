"""
Item Understand Pretrain Task
Input: caption parquet (pid, dense_caption) + pid2sid parquet
Output: LLM Pretrain format parquet (segments)

Task: Build pretrain data with SID and caption using various templates.
"""

import pandas as pd
import argparse
import json
import uuid
import random
from pathlib import Path
from tqdm import tqdm

# ============== Configuration ==============
SID_FORMAT = '<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>'

# Pretrain format templates
PRETRAIN_TEMPLATES = [
    # Format 1: JSON format
    lambda sid, caption: json.dumps({"视频ID": sid, "视频内容": caption}, ensure_ascii=False),
    # Format 2: Display format
    lambda sid, caption: f"视频{sid} 展示了以下内容：{caption}",
    # Format 3: Full description format
    lambda sid, caption: f"视频{sid} 的内容完整描述如下：{caption}",
]


# ============== Core Functions ==============
def pid_to_sid(pid, pid2sid: dict) -> str:
    """Convert a single pid to SID string."""
    if pid not in pid2sid:
        return ""
    code = pid2sid[pid]
    return SID_FORMAT.format(c0=code[0], c1=code[1], c2=code[2])


def build_segments(sid: str, caption: str) -> str:
    """Build segments format JSON string for pretrain."""
    template = random.choice(PRETRAIN_TEMPLATES)
    text = template(sid, caption)
    segments = [{"type": "text", "text": text}]
    return json.dumps(segments, ensure_ascii=False)


def process_row(row, pid2sid: dict) -> dict:
    """Process a single row of data."""
    pid = row['pid']
    dense_caption = row['dense_caption']

    # Check data validity
    if dense_caption is None or (isinstance(dense_caption, float) and pd.isna(dense_caption)):
        return None
    if not dense_caption:
        return None

    # Convert pid to SID
    sid = pid_to_sid(pid, pid2sid)
    if not sid:
        return None

    return {
        'source': 'RecIF_ItemUnderstand_Pretrain',
        'uuid': str(uuid.uuid4()),
        'segments': build_segments(sid, dense_caption),
        'metadata': json.dumps({'pid': int(pid), 'sid': sid}, ensure_ascii=False)
    }


# ============== Main Function ==============
def main():
    parser = argparse.ArgumentParser(description="Item Understand Pretrain Data Processing")
    parser.add_argument('--input', type=str, required=True, help='Input caption parquet path')
    parser.add_argument('--pid2sid', type=str, required=True, help='pid2sid mapping parquet path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load pid2sid mapping
    print(f"Loading pid2sid from {args.pid2sid}...")
    df_pid2sid = pd.read_parquet(args.pid2sid)
    pid2sid = dict(zip(df_pid2sid['pid'], df_pid2sid['sid']))
    print(f"  Loaded {len(pid2sid):,} mappings")

    # 2. Load caption data
    print(f"Loading caption data from {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"  Loaded {len(df):,} rows")

    # 3. Process data
    print("Processing...")
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        result = process_row(row, pid2sid)
        if result:
            results.append(result)

    # 4. Save results
    df_output = pd.DataFrame(results)
    output_path = output_dir / 'train.parquet'
    df_output.to_parquet(output_path, index=False)

    print(f"Saved: {output_path} ({len(df_output):,} rows)")
    print("Done!")


if __name__ == "__main__":
    main()
