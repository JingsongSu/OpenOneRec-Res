"""
Label Prediction Task (Point-wise Classification)
Input: metadata parquet + pid2sid parquet
Output: LLM SFT training format parquet

Task: Predict whether a user will "longview" (watch for a long time) a candidate video.
Binary classification: "是" (yes) or "否" (no).
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
INTERACTION_MAX_LEN = 10
TARGET_MAX_LEN = 10

# Interaction types
INTERACTION_TYPES = ["longview", "like", "follow", "forward", "not_interested"]

# System prompts (Chinese)
SYSTEM_PROMPTS = [
    "你是一个内容推荐专家，擅长分析用户的互动模式，预测用户的内容偏好。",
    "你是一个个性化推荐系统，能够基于用户的历史互动行为，预测用户未来可能产生的互动。",
    "你是一个用户行为分析助手，专注于理解用户的兴趣偏好，并推荐相关内容。",
    "你是一个内容推荐引擎，通过学习用户的互动历史，预测用户对新内容的反应。",
]

# Interaction type descriptions (Chinese)
INTERACTION_PROMPTS = {
    "longview": ["用户长时观看过以下内容：", "用户完整观看过的内容：", "用户深度浏览过以下内容："],
    "like": ["用户点赞过以下内容：", "用户喜欢的内容：", "获得用户点赞的内容："],
    "follow": ["用户关注过以下内容的作者：", "用户关注了这些内容的创作者："],
    "forward": ["用户转发过以下内容：", "用户分享过的内容：", "用户向他人推荐的内容："],
    "not_interested": ["用户表示不感兴趣的内容：", "用户标记为不感兴趣的内容："],
}

# Classification question prompts (Chinese)
CLASSIFICATION_QUESTIONS = [
    "请判断用户是否会长时观看视频{candidate_sid}？",
    "用户会完整观看视频{candidate_sid}吗？",
    "预测用户是否会深度观看视频{candidate_sid}。",
    "视频{candidate_sid}能够吸引用户长时间观看吗？",
    "用户会花时间仔细观看视频{candidate_sid}吗？",
]

# Classification answers
POSITIVE_ANSWER = "是"
NEGATIVE_ANSWER = "否"


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


def pid_to_sid(pid, pid2sid: dict) -> str:
    """Convert a single pid to SID string."""
    if pid in pid2sid:
        code = pid2sid[pid]
        return SID_FORMAT.format(c0=code[0], c1=code[1], c2=code[2])
    return ""


def build_messages(user_content: str, question: str, answer: str) -> str:
    """Build messages format JSON string."""
    system_prompt = random.choice(SYSTEM_PROMPTS)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_content + "\n" + question}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    ]
    return json.dumps(messages, ensure_ascii=False)


def process_row(row, pid2sid: dict) -> list:
    """Process a single row of data. Returns a list of results (one per candidate video)."""
    hist_pids = row['hist_video_pid']
    target_pids = row['target_video_pid']

    # Check data validity
    if hist_pids is None or (isinstance(hist_pids, float) and pd.isna(hist_pids)):
        return []
    if target_pids is None or (isinstance(target_pids, float) and pd.isna(target_pids)):
        return []
    if len(target_pids) == 0:
        return []

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
        return []

    # Get target longview mask
    target_longview_col = 'target_video_longview'
    if target_longview_col not in row or row[target_longview_col] is None:
        return []

    target_longview_mask = row[target_longview_col]
    if isinstance(target_longview_mask, float) and pd.isna(target_longview_mask):
        return []

    if len(target_longview_mask) != len(target_pids):
        return []

    # Limit target candidates
    limited_target_pids = target_pids[:TARGET_MAX_LEN]
    limited_longview_mask = target_longview_mask[:TARGET_MAX_LEN]

    # Build user content
    user_content = "\n".join(user_content_parts)

    # Generate one sample per candidate video
    results = []
    for candidate_pid, label in zip(limited_target_pids, limited_longview_mask):
        label = int(label)

        # Convert candidate pid to SID
        candidate_sid = pid_to_sid(candidate_pid, pid2sid)
        if not candidate_sid:
            continue

        # Build question with candidate SID
        question = random.choice(CLASSIFICATION_QUESTIONS).format(candidate_sid=candidate_sid)

        # Determine answer based on label
        answer = POSITIVE_ANSWER if label == 1 else NEGATIVE_ANSWER

        result = {
            'source': 'RecIF_LabelPred',
            'uuid': str(uuid.uuid4()),
            'messages': build_messages(user_content, question, answer),
            'metadata': json.dumps({
                'uid': int(row['uid']),
                'label': label
            }, ensure_ascii=False)
        }
        results.append(result)

    return results


# ============== Main Function ==============
def main():
    parser = argparse.ArgumentParser(description="Label Prediction Task Data Processing")
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
    positive_count, negative_count = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        if row['split'] != 0:
            continue
        row_results = process_row(row, pid2sid)
        for result in row_results:
            metadata = json.loads(result['metadata'])
            label = metadata['label']
            results.append(result)
            if label == 1:
                positive_count += 1
            else:
                negative_count += 1

    # 4. Save results
    df_train = pd.DataFrame(results)
    train_path = output_dir / 'train.parquet'
    df_train.to_parquet(train_path, index=False)

    print(f"Saved: {train_path} ({len(df_train):,} rows, pos={positive_count:,}, neg={negative_count:,})")
    print("Done!")


if __name__ == "__main__":
    main()
