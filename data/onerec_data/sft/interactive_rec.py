"""
Interactive Recommendation Task
Input: metadata parquet + pid2sid parquet
Output: LLM SFT training format parquet

Task: Given user profile (inter_user_profile_with_sid) and search keyword,
predict items the user will interact with.
"""

import pandas as pd
import json
import uuid
import random
import argparse
from pathlib import Path
from tqdm import tqdm

# ============== Configuration ==============
SID_FORMAT = '<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>'
TARGET_MAX_LEN = 10

# System prompts (Chinese)
SYSTEM_PROMPTS = [
    "你是一个智能推荐助手，能够根据用户的兴趣画像和当前对话需求，精准推荐用户可能感兴趣的内容。",
    "你是一个个性化推荐专家，擅长理解用户画像和对话意图，提供精准的内容推荐。",
    "你是一个对话式推荐系统，能够基于用户的兴趣特征和搜索意图，推荐最相关的内容。",
    "你是一个交互式内容推荐引擎，专注于理解用户画像和对话上下文，提供个性化推荐。",
    "你是一个智能内容顾问，通过分析用户兴趣和对话关键词，推荐符合需求的内容。",
    "你是一位资深的推荐算法专家，精通用户画像分析和个性化匹配，能够为每位用户提供量身定制的内容推荐。",
    "你是一个具备深度学习能力的推荐引擎，可以准确捕捉用户兴趣点，并结合实时需求给出最优推荐方案。",
    "你是一个智能化的内容匹配系统，擅长从海量信息中筛选出与用户画像和查询意图高度契合的内容。",
    "你是一个AI驱动的推荐助理，能够综合分析用户的历史偏好和当前需求，提供精准且多元化的内容推荐。",
    "你是一个智慧型推荐顾问，通过理解用户的兴趣图谱和语义意图，实现千人千面的个性化推荐。",
]

# User prompts (Chinese)
USER_PROMPTS = [
    "用户画像：\n{user_profile}\n\n用户查询：{keyword}\n\n请推荐相关内容。",
    "用户兴趣：\n{user_profile}\n\n搜索关键词：{keyword}\n\n请根据用户需求推荐内容。",
    "用户特征：\n{user_profile}\n\n当前需求：{keyword}\n\n请提供个性化推荐。",
    "【用户画像】\n{user_profile}\n\n【用户输入】\n{keyword}\n\n基于以上信息，推荐合适的内容。",
    "用户的兴趣偏好：\n{user_profile}\n\n用户正在寻找：{keyword}\n\n请推荐最相关的内容。",
    "这是用户的兴趣画像：\n{user_profile}\n\n用户现在想了解关于\"{keyword}\"的内容，能帮忙推荐一些吗？",
    "用户平时喜欢：\n{user_profile}\n\n现在用户搜索了\"{keyword}\"，请根据用户的兴趣推荐相关内容。",
    "根据用户画像显示，用户兴趣如下：\n{user_profile}\n\n用户刚刚输入了\"{keyword}\"，麻烦推荐一些合适的内容。",
    "用户的兴趣领域包括：\n{user_profile}\n\n用户正在查找\"{keyword}\"相关的内容，请给出推荐。",
    "用户的个人画像如下：\n{user_profile}\n\n用户搜索了\"{keyword}\"这个关键词，请推荐一些相关的内容。",
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


def build_messages(user_profile: str, keyword: str, answer: str) -> str:
    """Build messages format JSON string."""
    system_prompt = random.choice(SYSTEM_PROMPTS)
    user_prompt = random.choice(USER_PROMPTS).format(user_profile=user_profile, keyword=keyword)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    ]
    return json.dumps(messages, ensure_ascii=False)


def process_row(row, pid2sid: dict) -> list:
    """Process a single row of data. Returns a list of results (one per keyword)."""
    user_profile = row.get('inter_user_profile_with_sid')
    inter_keyword_to_items = row['inter_keyword_to_items']

    # Check user profile validity
    if user_profile is None or (isinstance(user_profile, float) and pd.isna(user_profile)):
        return []
    if not user_profile or not isinstance(user_profile, str):
        return []

    # Check keyword_to_items validity
    if inter_keyword_to_items is None or (isinstance(inter_keyword_to_items, float) and pd.isna(inter_keyword_to_items)):
        return []

    # Parse JSON string if needed
    if isinstance(inter_keyword_to_items, str):
        try:
            inter_keyword_to_items = json.loads(inter_keyword_to_items)
        except json.JSONDecodeError:
            return []

    if not isinstance(inter_keyword_to_items, dict) or len(inter_keyword_to_items) == 0:
        return []

    results = []
    for keyword, item_ids in inter_keyword_to_items.items():
        if not keyword or not item_ids:
            continue

        # Convert target items to SIDs
        answer = pids_to_sids(item_ids[:TARGET_MAX_LEN], pid2sid)
        if not answer:
            continue

        result = {
            'source': 'RecIF_InteractiveRec',
            'uuid': str(uuid.uuid4()),
            'messages': build_messages(user_profile, keyword, answer),
            'metadata': json.dumps({'uid': int(row['uid']), 'keyword': keyword}, ensure_ascii=False)
        }
        results.append(result)

    return results


# ============== Main Function ==============
def main():
    parser = argparse.ArgumentParser(description="Interactive Recommendation Task Data Processing")
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
        row_results = process_row(row, pid2sid)
        for result in row_results:
            results.append(result)

    # 4. Save results
    df_train = pd.DataFrame(results)
    train_path = output_dir / 'train.parquet'
    df_train.to_parquet(train_path, index=False)

    print(f"Saved: {train_path} ({len(df_train):,} rows)")
    print("Done!")


if __name__ == "__main__":
    main()
