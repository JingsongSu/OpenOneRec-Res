from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("VLLM_USE_V1", "0")

from vllm import LLM

from common import (
    load_jsonl,
    model_layout,
    parse_pooling_output,
    records_with_tokens,
    tokenize_prompts,
    tokenizer_for,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--request_batch_size", type=int, default=64)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--enforce_eager", action="store_true")
    args = parser.parse_args()

    layout = model_layout(args.model)
    tokenizer = tokenizer_for(args.model)
    rows = load_jsonl(args.input_jsonl)
    prompts = tokenize_prompts(
        rows,
        tokenizer,
        layout["sid_begin_token_id"],
    )

    llm = LLM(
        model=args.model,
        task="embed",
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
    )

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for start in range(0, len(rows), args.request_batch_size):
            row_batch = rows[start : start + args.request_batch_size]
            prompt_batch = prompts[
                start : start + args.request_batch_size
            ]
            outputs = llm.encode(prompt_batch, use_tqdm=False)
            for row, output in zip(row_batch, outputs):
                candidates = parse_pooling_output(
                    output,
                    beam_size=layout["beam_size"],
                )
                handle.write(
                    json.dumps(
                        {
                            "id": row.get("id", ""),
                            "candidates": records_with_tokens(
                                candidates,
                                tokenizer,
                                layout["layer_starts"],
                            ),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(f"Wrote {len(rows)} predictions to {output_path}")


if __name__ == "__main__":
    main()
