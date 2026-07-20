"""
Video Recommendation Task
Input: metadata parquet + pid2sid parquet
Output: LLM SFT training format parquet
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
SID_FORMAT = '<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>'
HIST_MAX_LEN = 512
TARGET_MAX_LEN = 10

# System prompts (Chinese)
SYSTEM_PROMPTS = [
    "你是一个智能推荐助手，能够根据用户的浏览历史预测用户可能感兴趣的下一个内容。",
    "你是一名内容推荐专家，擅长分析用户浏览行为并预测用户偏好。",
    "作为推荐系统助手，你需要根据用户历史浏览记录推荐合适的内容。",
    "你具备理解用户浏览模式并生成个性化推荐的能力。",
    "你是一个专业的内容推荐助手，能够根据用户过往浏览记录推荐相关内容。",
]

# User prompts (Chinese)
USER_PROMPTS = [
    "根据以下用户浏览记录，请预测用户接下来可能观看的内容：\n{query}",
    "用户浏览了以下内容：\n{query}\n请预测用户的下一个观看意向。",
    "以下是用户的浏览历史：\n{query}\n请推荐用户可能感兴趣的下一个内容。",
    "用户历史浏览记录如下：\n{query}\n分析并预测用户接下来会观看什么内容。",
    "{query}\n根据上述浏览记录，推测用户的下一个观看目标。",
]


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


def build_messages(query: str, answer: str) -> str:
    """Build messages format JSON string."""
    system_prompt = random.choice(SYSTEM_PROMPTS)
    user_prompt = random.choice(USER_PROMPTS).format(query=query)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    ]
    return json.dumps(messages, ensure_ascii=False)


def process_row(row, pid2sid: dict) -> dict:
    """Process a single row of data."""
    hist_pids = row['hist_video_pid']
    target_pids = row['target_video_pid']

    # Check data validity
    if hist_pids is None or (isinstance(hist_pids, float) and pd.isna(hist_pids)):
        return None
    if target_pids is None or (isinstance(target_pids, float) and pd.isna(target_pids)):
        return None

    # Truncate and convert to SID (keep most recent history)
    query = pids_to_sids(hist_pids[-HIST_MAX_LEN:], pid2sid)
    answer = pids_to_sids(target_pids[:TARGET_MAX_LEN], pid2sid)

    if not query or not answer:
        return None

    return {
        'source': 'RecIF_VideoRec',
        'uuid': str(uuid.uuid4()),
        'messages': build_messages(query, answer),
        'metadata': json.dumps({'uid': int(row['uid'])}, ensure_ascii=False)
    }


# ============== Main Function ==============
def main():
    parser = argparse.ArgumentParser(description="Video Recommendation Task Data Processing")
    parser.add_argument('--input', type=str, required=True, help='Input metadata parquet path')
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
