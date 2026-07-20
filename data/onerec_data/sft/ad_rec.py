"""
Ad Recommendation Task (Cross-domain)
Input: metadata parquet + pid2sid parquet
Output: LLM SFT training format parquet

Task: Predict ad videos the user will click based on video watch history and ad click history.
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
VIDEO_HIST_MAX_LEN = 100
AD_HIST_MAX_LEN = 200
TARGET_MAX_LEN = 10

# System prompts (Chinese)
SYSTEM_PROMPTS = [
    "你是一个智能广告推荐助手，能够根据用户的视频观看历史和广告点击行为，预测用户接下来可能点击的广告视频。",
    "你是一个广告点击预测专家，擅长分析用户的观看习惯和广告点击偏好，预测用户的广告兴趣。",
    "你是一个个性化广告推荐系统，能够基于用户的视频观看历史和广告点击记录，预测用户未来可能点击的广告。",
    "你是一个用户行为分析助手，专注于理解用户的内容偏好和广告兴趣，推荐相关广告视频。",
    "你是一个广告推荐引擎，通过学习用户的视频观看和广告点击历史，预测用户对广告的兴趣。",
]

# Video watch history prompts (Chinese)
VIDEO_WATCH_PROMPTS = [
    "用户观看过的视频：",
    "用户浏览过的视频内容：",
    "用户长时间观看的视频：",
    "用户感兴趣的视频：",
]

# Ad click history prompts (Chinese)
AD_CLICK_PROMPTS = [
    "用户点击过的广告视频：",
    "用户浏览过的广告视频：",
    "用户感兴趣的广告视频：",
    "用户历史广告点击记录：",
]

# Task prompts (Chinese)
TASK_PROMPTS = [
    "请根据用户的观看和广告点击历史，预测用户接下来可能点击的广告视频。",
    "基于以上记录，推荐用户可能感兴趣并点击的广告视频。",
    "分析用户的行为偏好，预测用户下一步会点击哪些广告视频。",
    "根据用户的视频观看和广告点击习惯，推荐用户可能点击的广告视频。",
    "请推荐用户接下来可能感兴趣并点击的广告视频。",
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
    hist_ad_pids = row['hist_ad_pid']
    target_ad_pids = row['target_ad_pid']

    # Check data validity
    if target_ad_pids is None or (isinstance(target_ad_pids, float) and pd.isna(target_ad_pids)):
        return None
    if len(target_ad_pids) == 0:
        return None

    # Build user content parts
    user_content_parts = []

    # 1. Process video watch history (long-view videos)
    hist_longview_video_list = row['hist_longview_video_list']
    if hist_longview_video_list is not None and not (isinstance(hist_longview_video_list, float) and pd.isna(hist_longview_video_list)):
        if len(hist_longview_video_list) > 0:
            # Keep the most recent videos (rightmost in the list)
            video_sids = pids_to_sids(hist_longview_video_list[-VIDEO_HIST_MAX_LEN:], pid2sid)
            if video_sids:
                video_prompt = random.choice(VIDEO_WATCH_PROMPTS)
                user_content_parts.append(f"{video_prompt}{video_sids}")

    # 2. Process ad click history
    if hist_ad_pids is not None and not (isinstance(hist_ad_pids, float) and pd.isna(hist_ad_pids)):
        if len(hist_ad_pids) > 0:
            # Keep the most recent ads (rightmost in the list)
            ad_sids = pids_to_sids(hist_ad_pids[-AD_HIST_MAX_LEN:], pid2sid)
            if ad_sids:
                ad_prompt = random.choice(AD_CLICK_PROMPTS)
                user_content_parts.append(f"{ad_prompt}{ad_sids}")

    # Need at least one type of history
    if not user_content_parts:
        return None

    # 3. Process target ad videos
    answer = pids_to_sids(target_ad_pids[:TARGET_MAX_LEN], pid2sid)
    if not answer:
        return None

    # Build final messages
    user_content = "\n".join(user_content_parts)
    task_prompt = random.choice(TASK_PROMPTS)

    return {
        'source': 'RecIF_AdRec',
        'uuid': str(uuid.uuid4()),
        'messages': build_messages(user_content, task_prompt, answer),
        'metadata': json.dumps({'uid': int(row['uid'])}, ensure_ascii=False)
    }


# ============== Main Function ==============
def main():
    parser = argparse.ArgumentParser(description="Ad Recommendation Task Data Processing")
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
