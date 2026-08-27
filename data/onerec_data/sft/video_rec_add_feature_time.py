"""
Video Recommendation SFT Task with Add-Feature + Time Token

Input: metadata parquet + pid2sid parquet + mid2sid parquet
Output: LLM SFT training format parquet

Task:
- Query/history: MID token + (ctype + SID + relative-time token) for each historical item
- Answer/target: (ctype + SID) only
- Relative time is measured from each history event to the first target event.
"""

import pandas as pd
import numpy as np
import argparse
import json
import uuid
import random
from pathlib import Path
from tqdm import tqdm


# ============== Configuration ==============
MID_SID_FORMAT = '<mid_a_{t0}><mid_b_{t1}><mid_c_{t2}>'

# 删除 ls 特征，并将 ctype 放在 sid_begin 前面
SID_FORMAT = '<|ctype_{k2}|><|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><s_d_{c3}><|sid_end|>'

# 0-336 hours; >336 hours is clipped to 336
TIME_FORMAT = '<|time_{t0}|>'

HIST_MAX_LEN = 512
TARGET_MAX_LEN = 10


# System prompts (Chinese)
SYSTEM_PROMPTS = [
    "你是一个智能推荐助手，能够根据用户的历史行为预测用户可能感兴趣的下一个内容。",
#     "你是一名内容推荐专家，擅长分析用户行为并预测用户偏好。",
#     "作为推荐系统助手，你需要根据用户历史点击推荐合适的内容。",
#     "你具备理解用户行为模式并生成个性化推荐的能力。",
#     "你是一个专业的内容推荐助手，能够根据用户过往行为记录推荐相关内容。",
]

# User prompts (Chinese)
USER_PROMPTS = [
    "根据以下用户历史行为序列，请预测用户接下来可能点击的广告：\n{query}",
#     "用户浏览了以下广告：\n{query}\n请预测用户的下一个点击意向。",
#     "以下是用户的点击的广告序列：\n{query}\n请推荐用户可能感兴趣的下一个广告。",
#     "用户依次点击了以下：\n{query}\n分析并预测用户接下来会点击的广告。",
#     "{query}\n根据上述浏览记录，推测用户的下一个点击目标。",
]


# ============== Core Functions ==============
def pids_to_sids(pids, ctypeids, pid2sid: dict) -> str:
    """Convert a list of pids + ctype ids to SID string."""
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
    """Convert pids to MID + ctype + SID string (without time token)."""
    if pids is None or (isinstance(pids, float) and pd.isna(pids)):
        return ""

    sids = []

    # mid 命中码本时添加 MID token；
    # mid 未命中时不添加，但继续生成后续 item token
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


def pids_to_sids_add_feature_time(
    mid,
    pids,
    ctypeids,
    pid2sid: dict,
    mid2sid: dict,
    relative_time
) -> str:
    """
    Convert history to:
        MID + (ctype + SID + time) * history_len

    The time token is appended immediately after the corresponding item SID.
    """
    if pids is None or (isinstance(pids, float) and pd.isna(pids)):
        return ""

    sids = []

    # MID token is added only once, at the beginning of the history sequence.
    if mid != "null" and mid in mid2sid:
        m_code = mid2sid[mid]

        m_sid = MID_SID_FORMAT.format(
            t0=m_code[0],
            t1=m_code[1],
            t2=m_code[2]
        )

        sids.append(m_sid)

    for pid, ctype, t in zip(pids, ctypeids, relative_time):
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

            tm = TIME_FORMAT.format(t0=t)
            sids.append(tm)

    return ''.join(sids)


def build_messages(query: str, answer: str) -> str:
    """Build messages format JSON string."""
    system_prompt = random.choice(SYSTEM_PROMPTS)
    user_prompt = random.choice(USER_PROMPTS).format(query=query)

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_prompt
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": answer
                }
            ]
        }
    ]

    return json.dumps(messages, ensure_ascii=False)


def bucketization_fn(hist_time, target_time, num_buckets=336):
    """
    Convert target-relative time difference to hour-level buckets.

    time_bucket = min((target_time - hist_time) // 3600, 336)
    """
    rel_time = target_time - hist_time
    rel_hours = np.minimum(rel_time // 3600, num_buckets)
    return rel_hours


def process_row(row, pid2sid: dict, mid2sid: dict) -> dict:
    """Process a single row of data."""
    mid = row["mid"]

    hist_pids = row['hist_adid']
    target_pids = row['target_adid']

    hist_ctypeids = row['hist_ctypeid']
    target_ctypeids = row['target_ctypeid']

    hist_adid_time = row['hist_adid_time']
    target_adid_time = row['target_adid_time'][0]

    relative_time = bucketization_fn(
        hist_adid_time,
        target_adid_time
    )

    # Check data validity
    if hist_pids is None or (
        isinstance(hist_pids, float) and pd.isna(hist_pids)
    ):
        return None

    if target_pids is None or (
        isinstance(target_pids, float) and pd.isna(target_pids)
    ):
        return None

    # Query/history:
    # MID + (ctype + SID + time), keep most recent HIST_MAX_LEN items
    query = pids_to_sids_add_feature_time(
        mid,
        hist_pids[-HIST_MAX_LEN:],
        hist_ctypeids[-HIST_MAX_LEN:],
        pid2sid,
        mid2sid,
        relative_time[-HIST_MAX_LEN:]
    )

    # Answer/target:
    # Keep the original add_feature behavior; target does NOT contain time tokens
    answer = pids_to_sids_add_feature(
        "null",
        target_pids[:TARGET_MAX_LEN],
        target_ctypeids[:TARGET_MAX_LEN],
        pid2sid,
        mid2sid
    )

    if not query or not answer:
        return None

    return {
        'source': 'AdRec_SFT',
        'uuid': str(uuid.uuid4()),
        'messages': build_messages(query, answer),
        'metadata': json.dumps(
            {'mid': row['mid']},
            ensure_ascii=False
        )
    }


# ============== Main Function ==============
def main():
    parser = argparse.ArgumentParser(
        description="Video Recommendation Task Data Processing"
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

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    random.seed(args.seed)

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

    # 3. Process data (train only, split=0)
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
