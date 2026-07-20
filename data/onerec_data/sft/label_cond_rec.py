"""
Label Conditional Recommendation Task
Input: metadata parquet + pid2sid parquet
Output: LLM SFT training format parquet

Task: Predict items that users will interact with under specific behavior types
(longview/like/follow/forward/not_interested).
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
INTERACTION_MAX_LEN = 10  # Max items per interaction type

# Interaction types
INTERACTION_TYPES = ["longview", "like", "follow", "forward", "not_interested"]

# System prompts (Chinese)
SYSTEM_PROMPTS = [
    "你是一个智能推荐助手，能够根据用户对不同内容的互动行为，精准推荐用户可能感兴趣的下一个内容。",
    "你是一个内容推荐专家，擅长分析用户的互动模式，预测用户的内容偏好。",
    "你是一个个性化推荐系统，能够基于用户的历史互动行为，预测用户未来可能产生的互动。",
    "你是一个用户行为分析助手，专注于理解用户的兴趣偏好，并推荐相关内容。",
]

# Interaction type descriptions (Chinese)
INTERACTION_PROMPTS = {
    "longview": ["用户长时观看过以下内容：", "用户完整观看过的内容：", "用户深度浏览过以下内容："],
    "like": ["用户点赞过以下内容：", "用户喜欢的内容：", "获得用户点赞的内容："],
    "follow": ["用户关注过以下内容的作者：", "用户关注了这些内容的创作者："],
    "forward": ["用户转发过以下内容：", "用户分享过的内容：", "用户向他人推荐的内容："],
    "not_interested": ["用户表示不感兴趣的内容：", "用户标记为不感兴趣的内容："],
}

# Task prompts for each interaction type (Chinese)
TASK_PROMPTS = {
    "longview": ["请根据用户的互动行为，推荐用户可能会长时观看的内容。", "基于以上互动记录，预测用户会完整观看的内容。"],
    "like": ["请根据用户的互动行为，推荐用户可能会点赞的内容。", "基于用户的互动偏好，预测用户会给哪些内容点赞。"],
    "follow": ["请根据用户的互动行为，推荐用户可能会关注其作者的内容。", "基于用户的关注偏好，预测用户会关注哪些内容的创作者。"],
    "forward": ["请根据用户的互动行为，推荐用户可能会转发的内容。", "基于用户的分享习惯，预测用户会转发的内容。"],
    "not_interested": ["请根据用户的互动行为，预测用户可能会表示不感兴趣的内容。", "基于用户的偏好，预测用户可能会标记为不感兴趣的内容。"],
}


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


def build_messages(user_content: str, task_prompt: str, answer: str) -> str:
    """Build messages format JSON string."""
    system_prompt = random.choice(SYSTEM_PROMPTS)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_content + "\n" + task_prompt}]},
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

    # Build user interaction history description
    user_content_parts = []
    for interaction in INTERACTION_TYPES:
        hist_col = f'hist_video_{interaction}'
        if hist_col not in row or row[hist_col] is None:
            continue

        mask = row[hist_col]
        if isinstance(mask, float) and pd.isna(mask):
            continue

        # Filter pids with interaction based on mask
        if len(mask) == len(hist_pids):
            mask_array = np.array(mask)
            pids_array = np.array(hist_pids)
            interaction_pids = pids_array[mask_array == 1].tolist()
            interaction_pids = interaction_pids[-INTERACTION_MAX_LEN:]

            if interaction_pids:
                sids = pids_to_sids(interaction_pids, pid2sid)
                if sids:
                    prompt = random.choice(INTERACTION_PROMPTS[interaction])
                    user_content_parts.append(f"{prompt}{sids}")

    if not user_content_parts:
        return None

    # Randomly select a target interaction type with data
    available_targets = []
    for interaction in INTERACTION_TYPES:
        target_col = f'target_video_{interaction}'
        if target_col not in row or row[target_col] is None:
            continue

        target_mask = row[target_col]
        if isinstance(target_mask, float) and pd.isna(target_mask):
            continue

        if len(target_mask) == len(target_pids) and sum(1 for x in target_mask if x == 1) > 0:
            available_targets.append(interaction)

    if not available_targets:
        return None

    # Randomly select an interaction type as target
    selected_interaction = random.choice(available_targets)
    target_col = f'target_video_{selected_interaction}'
    target_mask = row[target_col]

    # Filter target_pids
    target_mask_array = np.array(target_mask)
    target_pids_array = np.array(target_pids)
    filtered_target_pids = target_pids_array[target_mask_array == 1].tolist()
    filtered_target_pids = filtered_target_pids[:INTERACTION_MAX_LEN]

    # Convert to SID
    answer = pids_to_sids(filtered_target_pids, pid2sid)
    if not answer:
        return None

    # Build final messages
    user_content = "\n".join(user_content_parts)
    task_prompt = random.choice(TASK_PROMPTS[selected_interaction])

    return {
        'source': 'RecIF_LabelCondRec',
        'uuid': str(uuid.uuid4()),
        'messages': build_messages(user_content, task_prompt, answer),
        'metadata': json.dumps({'uid': int(row['uid']), 'target_interaction': selected_interaction}, ensure_ascii=False)
    }


# ============== Main Function ==============
def main():
    parser = argparse.ArgumentParser(description="Label Conditional Recommendation Task Data Processing")
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
