"""
User Profile Pretrain Task
Input: metadata parquet
Output: LLM Pretrain format parquet (segments)

Task: Directly use inter_user_profile_with_sid as pretrain text.
"""

import pandas as pd
import argparse
import json
import uuid
from pathlib import Path
from tqdm import tqdm


def process_row(row) -> dict:
    """Process a single row of data."""
    user_profile = row.get('inter_user_profile_with_sid')

    # Check data validity
    if user_profile is None or (isinstance(user_profile, float) and pd.isna(user_profile)):
        return None
    if not user_profile or not isinstance(user_profile, str):
        return None

    segments = [{"type": "text", "text": user_profile}]

    return {
        'source': 'RecIF_UserProfile_Pretrain',
        'uuid': str(uuid.uuid4()),
        'segments': json.dumps(segments, ensure_ascii=False),
        'metadata': json.dumps({'uid': int(row['uid'])}, ensure_ascii=False)
    }


def main():
    parser = argparse.ArgumentParser(description="User Profile Pretrain Data Processing")
    parser.add_argument('--input', type=str, required=True, help='Input metadata parquet path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    print(f"Loading metadata from {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"  Loaded {len(df):,} rows")

    # Process data (train only, split=0)
    print("Processing...")
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if row['split'] != 0:
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
