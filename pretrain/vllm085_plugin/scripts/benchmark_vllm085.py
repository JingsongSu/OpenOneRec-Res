"""Benchmark one method per process to avoid loading two engines together."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("VLLM_USE_V1", "0")

import numpy as np
from tqdm import tqdm
from vllm import LLM

from common import (
    layer_restricted_autoregressive_beam,
    load_jsonl,
    model_layout,
    parse_pooling_output,
    target_global_ids,
    tokenize_prompts,
    tokenizer_for,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["residual", "autoregressive"],
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--request_batch_size", type=int, default=64)
    parser.add_argument("--warmup_batches", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument(
        "--disable_prefix_caching",
        action="store_true",
    )
    return parser.parse_args()


def new_state() -> dict:
    return {
        "samples": 0,
        "hits": 0,
        "reciprocal_rank_sum": 0.0,
        "ndcg_sum": 0.0,
        "exact_top1": 0,
        "layer_top1_correct": [0, 0, 0, 0],
    }


def update_metrics(state: dict, candidates, target: list[int]) -> None:
    state["samples"] += 1
    target_tuple = tuple(target)
    rank = 0
    for index, candidate in enumerate(candidates, 1):
        if candidate.global_ids == target_tuple:
            rank = index
            break
    if rank:
        state["hits"] += 1
        state["reciprocal_rank_sum"] += 1.0 / rank
        state["ndcg_sum"] += 1.0 / math.log2(rank + 1)

    if candidates and candidates[0].global_ids == target_tuple:
        state["exact_top1"] += 1
    if candidates:
        top = candidates[0].global_ids
        for layer, expected in enumerate(target):
            if int(top[layer]) == int(expected):
                state["layer_top1_correct"][layer] += 1


def finalize(state: dict) -> dict:
    n = max(1, state["samples"])
    return {
        "samples": state["samples"],
        "recall_at_beam": state["hits"] / n,
        "mrr_at_beam": state["reciprocal_rank_sum"] / n,
        "ndcg_at_beam": state["ndcg_sum"] / n,
        "exact_at_1": state["exact_top1"] / n,
        "layer_top1_accuracy": [
            value / n for value in state["layer_top1_correct"]
        ],
    }


def create_engine(args, layout) -> LLM:
    common = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
    )
    if args.method == "residual":
        return LLM(task="embed", **common)
    return LLM(
        task="generate",
        enable_prefix_caching=not args.disable_prefix_caching,
        max_logprobs=max(20, 2 * layout["beam_size"]),
        hf_overrides={
            "architectures": [
                "Qwen3ForCausalLMIgnoreResidualSIDV085"
            ]
        },
        **common,
    )


def run_method(engine, method, prompt_batch, layout):
    if method == "residual":
        outputs = engine.encode(prompt_batch, use_tqdm=False)
        return [
            parse_pooling_output(
                output,
                beam_size=layout["beam_size"],
            )
            for output in outputs
        ]
    return layer_restricted_autoregressive_beam(
        engine,
        prompt_batch,
        layer_starts=layout["layer_starts"],
        layer_sizes=layout["layer_sizes"],
        beam_size=layout["beam_size"],
    )


def main() -> None:
    args = parse_args()
    layout = model_layout(args.model)
    tokenizer = tokenizer_for(args.model)
    rows = load_jsonl(args.eval_jsonl, args.max_samples)
    if not rows:
        raise ValueError("No evaluation rows were loaded.")

    prompts = tokenize_prompts(
        rows,
        tokenizer,
        layout["sid_begin_token_id"],
    )
    targets = [
        target_global_ids(row, tokenizer)
        for row in rows
    ]
    batches = [
        (
            prompts[start : start + args.request_batch_size],
            targets[start : start + args.request_batch_size],
        )
        for start in range(0, len(rows), args.request_batch_size)
    ]

    engine = create_engine(args, layout)

    # Warm-up is excluded from timed metrics.
    for prompt_batch, _ in batches[: args.warmup_batches]:
        run_method(engine, args.method, prompt_batch, layout)

    state = new_state()
    batch_latencies = []
    per_sample_ms = []
    total_start = time.perf_counter()
    for prompt_batch, target_batch in tqdm(
        batches,
        desc=args.method,
    ):
        batch_start = time.perf_counter()
        parsed = run_method(
            engine,
            args.method,
            prompt_batch,
            layout,
        )
        elapsed = time.perf_counter() - batch_start
        batch_latencies.append(elapsed)
        per_sample_ms.append(
            elapsed * 1000.0 / len(prompt_batch)
        )
        for candidates, target in zip(parsed, target_batch):
            update_metrics(state, candidates, target)

    total_seconds = time.perf_counter() - total_start
    report = {
        "method": args.method,
        "config": {
            **vars(args),
            "vllm_version": "0.8.5",
            "num_sid_layers": 4,
            "beam_size": layout["beam_size"],
            "layer_starts": layout["layer_starts"],
            "layer_sizes": layout["layer_sizes"],
            "autoregressive_baseline": (
                "Every SID layer is constrained to its own token range, "
                "expands all surviving beams, then globally keeps B."
            ),
            "timing_excludes": (
                "engine loading, tokenizer loading, JSON parsing, "
                "prompt tokenization, and warm-up"
            ),
        },
        **finalize(state),
        "total_timed_seconds": total_seconds,
        "throughput_samples_per_second": len(rows) / total_seconds,
        "mean_batch_latency_ms": (
            statistics.fmean(batch_latencies) * 1000.0
        ),
        "p50_batch_latency_ms": float(
            np.percentile(batch_latencies, 50) * 1000.0
        ),
        "p95_batch_latency_ms": float(
            np.percentile(batch_latencies, 95) * 1000.0
        ),
        "mean_ms_per_sample": statistics.fmean(per_sample_ms),
        "p50_ms_per_sample": float(
            np.percentile(per_sample_ms, 50)
        ),
        "p95_ms_per_sample": float(
            np.percentile(per_sample_ms, 95)
        ),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
