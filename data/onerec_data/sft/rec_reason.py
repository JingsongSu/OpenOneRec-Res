"""
Recommendation Reasoning Task
Input: rec_reason parquet (user_profile_with_sid, gsu_caption, target_caption, cot, etc.)
Output: LLM SFT training format parquet

Task: Given user profile, watch history captions, and target video caption,
generate reasoning for why the user would click the target video.
"""

import pandas as pd
import argparse
import json
import uuid
from pathlib import Path
from tqdm import tqdm

# ============== Configuration ==============
USER_PROMPT_TEMPLATE = """{user_profile}

[历史观看视频内容]
{gsu_caption}

[用户点击下一个视频内容]
{target_video_caption}

请在思考的时候分析总结用户兴趣，重点根据用户观看视频内容进行推理，给出下一个点击的理由及视频的基本内容，下一个点击视频需要与给定的一致，注意虽然给出了下一个点击的视频但应该体现出推理得到而不是直接知道的。
最后再用一段话输出精炼的推理过程。
生成格式严格按照两大部分，标题分别是：预测分析；精炼推理。
"""


# ============== Core Functions ==============
def build_messages(user_prompt: str, answer: str) -> str:
    """Build messages format JSON string."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    ]
    return json.dumps(messages, ensure_ascii=False)


def is_valid_str(val) -> bool:
    """Check if value is a valid non-empty string."""
    if val is None:
        return False
    if isinstance(val, float) and pd.isna(val):
        return False
    if isinstance(val, str) and val.strip():
        return True
    return False


def process_row(row) -> dict:
    """Process a single row of data."""
    user_profile = row.get('inter_user_profile_with_sid')
    gsu_caption = row.get('reco_gsu_caption')
    target_caption = row.get('reco_target_caption')
    answer = row.get('reco_cot')

    # Check data validity
    if not is_valid_str(user_profile):
        return None
    if not is_valid_str(target_caption):
        return None
    if not is_valid_str(answer):
        return None

    gsu_caption = str(gsu_caption) 

    # Build user prompt
    user_prompt = USER_PROMPT_TEMPLATE.format(
        user_profile=user_profile,
        gsu_caption=gsu_caption,
        target_video_caption=target_caption
    )

    metadata = {
        'uid': int(row['uid']) if 'uid' in row else None,
        'target_pid': int(row['target_pid']) if 'target_pid' in row else None,
    }

    return {
        'source': 'RecIF_RecoReason',
        'uuid': str(uuid.uuid4()),
        'messages': build_messages(user_prompt, answer),
        'metadata': json.dumps(metadata, ensure_ascii=False)
    }


# ============== Main Function ==============
def main():
    parser = argparse.ArgumentParser(description="Recommendation Reasoning Task Data Processing")
    parser.add_argument('--input', type=str, required=True, help='Input rec_reason parquet path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"  Loaded {len(df):,} rows")

    # Process data (train only, split=0)
    print("Processing...")
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if row.get('split', 0) != 0:
            continue
        result = process_row(row)
        if result:
            results.append(result)

    # Save results
    df_train = pd.DataFrame(results)
    train_path = output_dir / 'train.parquet'
    df_train.to_parquet(train_path, index=False)

    print(f"Saved: {train_path} ({len(df_train):,} rows)")
    print("Done!")


if __name__ == "__main__":
    main()
