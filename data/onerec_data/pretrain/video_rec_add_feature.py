"""
Video Recommendation Pretrain Task
Input: metadata parquet + pid2sid parquet + mid2sid parquet
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

MID_SID_FORMAT = '<mid_a_{t0}><mid_b_{t1}><mid_c_{t2}>'

# 删除 ls 特征，并将 ctype 放到 sid_begin 前面
SID_FORMAT = '<|ctype_{k2}|><|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><s_d_{c3}><|sid_end|>'

HIST_MAX_LEN = 512
TARGET_MAX_LEN = 10


# ============== Core Functions ==============

def pids_to_sids(pids, ctypeids, pid2sid: dict) -> str:
    """Convert a list of pids to SID string."""
    if pids is None or (isinstance(pids, float) and pd.isna(pids)):
        return ""

    sids = []

    for pid, ctype in zip(pids, ctypeids):
        if pid in pid2sid:
            code = pid2sid[pid]

            sid = SID_FORMAT.format(
                k2=ctype,
                c0=code[0],
                c1=code[1],
                c2=code[2],
                c3=code[3]
            )

            sids.append(sid)

    return ''.join(sids)


def pids_to_sids_add_feature(
    mid,
    pids,
    ctypeids,
    pid2sid: dict,
    mid2sid: dict
) -> str:
    """Convert a list of pids to SID string."""
    if pids is None or (isinstance(pids, float) and pd.isna(pids)):
        return ""

    sids = []

    # mid 命中码本时添加 MID token；
    # 未命中时不添加，但继续构建后续 item token
    if mid != "null" and mid in mid2sid:
        m_code = mid2sid[mid]

        m_sid = MID_SID_FORMAT.format(
            t0=m_code[0],
            t1=m_code[1],
            t2=m_code[2]
        )

        sids.append(m_sid)

    for pid, ctype in zip(pids, ctypeids):
        if pid in pid2sid:
            item_code = pid2sid[pid]

            sid = SID_FORMAT.format(
                k2=ctype,
                c0=item_code[0],
                c1=item_code[1],
                c2=item_code[2],
                c3=item_code[3]
            )

            sids.append(sid)

    return ''.join(sids)


def build_segments(hist_sids: str, target_sids: str) -> str:
    """Build segments format JSON string for pretrain."""
    text = f"{hist_sids}{target_sids}"
    segments = [{"type": "text", "text": text}]

    return json.dumps(segments, ensure_ascii=False)


def process_row(row, pid2sid: dict, mid2sid: dict) -> dict:
    """Process a single row of data."""

    mid = row["mid"]

    hist_pids = row['hist_adid']
    target_pids = row['target_adid']

    hist_ctypeids = row['hist_ctypeid']
    target_ctypeids = row['target_ctypeid']

    # Check data validity
    if hist_pids is None or (
        isinstance(hist_pids, float) and pd.isna(hist_pids)
    ):
        return None

    if target_pids is None or (
        isinstance(target_pids, float) and pd.isna(target_pids)
    ):
        return None

    # Truncate and convert to SID
    hist_sids = pids_to_sids_add_feature(
        mid,
        hist_pids[-HIST_MAX_LEN:],
        hist_ctypeids[-HIST_MAX_LEN:],
        pid2sid,
        mid2sid
    )

    target_sids = pids_to_sids_add_feature(
        "null",
        target_pids[:TARGET_MAX_LEN],
        target_ctypeids[:TARGET_MAX_LEN],
        pid2sid,
        mid2sid
    )

    if not hist_sids or not target_sids:
        return None

    return {
        'source': 'AdRec_Pretrain',
        'uuid': str(uuid.uuid4()),
        'segments': build_segments(hist_sids, target_sids),
        'metadata': json.dumps(
            {'mid': row['mid']},
            ensure_ascii=False
        )
    }


# ============== Main Function ==============

def main():
    parser = argparse.ArgumentParser(
        description="Video Recommendation Pretrain Data Processing"
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input metadata parquet path'
    )

    parser.add_argument(
        '--pid2sid',
        type=str,
        required=True,
        help='pid2sid mapping parquet path'
    )

    parser.add_argument(
        '--mid2sid',
        type=str,
        required=True,
        help='mid2sid mapping parquet path'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load pid2sid mapping
    print(f"Loading adid2sid from {args.pid2sid}...")

    df_pid2sid = pd.read_parquet(args.pid2sid)
    pid2sid = dict(zip(df_pid2sid['adid'], df_pid2sid['sid']))

    print(f"  Loaded {len(pid2sid):,} mappings")

    # Load mid2sid mapping
    print(f"Loading mid2sid from {args.mid2sid}...")

    df_mid2sid = pd.read_parquet(args.mid2sid)
    mid2sid = dict(zip(df_mid2sid['mid'], df_mid2sid['sid']))

    print(f"  Loaded {len(mid2sid):,} mappings")

    # 2. Load metadata
    print(f"Loading metadata from {args.input}...")

    df = pd.read_parquet(args.input)

    print(f"  Loaded {len(df):,} rows")

    # 3. Process data
    print("Processing...")

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
#         if row['split'] != 0:
#             continue

        result = process_row(
            row,
            pid2sid,
            mid2sid
        )

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
