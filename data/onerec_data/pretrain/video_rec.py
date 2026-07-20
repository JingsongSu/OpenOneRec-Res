"""
Video Recommendation Pretrain Task
Input: metadata parquet + pid2sid parquet
Output: LLM Pretrain format parquet (segments instead of messages)

Task: Directly concatenate history SIDs and target SIDs without prompts.
"""

import pandas as pd
import argparse
import json
import uuid
from pathlib import Path
from tqdm import tqdm

# ============== Configuration ==============
SID_FORMAT = '<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>'
HIST_MAX_LEN = 512
TARGET_MAX_LEN = 10


# ============== Core Functions ==============
def pids_to_sids(pids, pid2sid: dict) -> str:
    """Convert a list of pids to SID string."""
    if pids is None or (isinstance(pids, float) and pd.isna(pids)):
        return ""
    sids = []
    for pid in pids:
        if pid in pid2sid:
            code = pid2sid[pid]
            sid = SID_FORMAT.format(c0=code[0], c1=code[1], c2=code[2])
            sids.append(sid)
    return ''.join(sids)


def build_segments(hist_sids: str, target_sids: str) -> str:
    """Build segments format JSON string for pretrain."""
    text = f"{hist_sids}{target_sids}"
    segments = [{"type": "text", "text": text}]
    return json.dumps(segments, ensure_ascii=False)


def process_row(row, pid2sid: dict) -> dict:
    """Process a single row of data."""
    hist_pids = row['hist_video_pid']
    target_pids = row['target_video_pid']

    # Check data validity
    if hist_pids is None or (isinstance(hist_pids, float) and pd.isna(hist_pids)):
        return None
    if target_pids is None or (isinstance(target_pids, float) and pd.isna(target_pids)):
        return None

    # Truncate and convert to SID
    hist_sids = pids_to_sids(hist_pids[-HIST_MAX_LEN:], pid2sid)
    target_sids = pids_to_sids(target_pids[:TARGET_MAX_LEN], pid2sid)

    if not hist_sids or not target_sids:
        return None

    return {
        'source': 'RecIF_VideoRec_Pretrain',
        'uuid': str(uuid.uuid4()),
        'segments': build_segments(hist_sids, target_sids),
        'metadata': json.dumps({'uid': int(row['uid'])}, ensure_ascii=False)
    }


# ============== Main Function ==============
def main():
    parser = argparse.ArgumentParser(description="Video Recommendation Pretrain Data Processing")
    parser.add_argument('--input', type=str, required=True, help='Input metadata parquet path')
    parser.add_argument('--pid2sid', type=str, required=True, help='pid2sid mapping parquet path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load pid2sid mapping
    print(f"Loading pid2sid from {args.pid2sid}...")
    df_pid2sid = pd.read_parquet(args.pid2sid)
    pid2sid = dict(zip(df_pid2sid['pid'], df_pid2sid['sid']))
    print(f"  Loaded {len(pid2sid):,} mappings")

    # 2. Load metadata
    print(f"Loading metadata from {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"  Loaded {len(df):,} rows")

    # 3. Process data (train only, split=0)
    print("Processing...")
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if row['split'] != 0:
            continue
        result = process_row(row, pid2sid)
        if result:
            results.append(result)

    # 4. Save results
    df_train = pd.DataFrame(results)
    train_path = output_dir / 'train.parquet'
    df_train.to_parquet(train_path, index=False)

    print(f"Saved: {train_path} ({len(df_train):,} rows)")
    print("Done!")


if __name__ == "__main__":
    main()
