#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenOneRec-Res residual SID 评测脚本，对应“MID 只出现一次”的数据格式。

期望 SFT 数据：

user:
    指令文本
    <mid_a_*><mid_b_*><mid_c_*>
    <history_ctype_1><|sid_begin|><history_sid_1><|sid_end|>
    <history_ctype_2><|sid_begin|><history_sid_2><|sid_end|>
    ...

assistant:
    <target_ctype><|sid_begin|><target_sid><|sid_end|>

推理时：
    1. system + user 通过 chat template 构造 prompt；
    2. MID 已经在 user prompt 中出现一次；
    3. 从 assistant 标签提取 target CType；
    4. 在 assistant generation header 后追加：
           target CType + SID_BEGIN
    5. prompt 最后一个 token 为 SID_BEGIN；
    6. 调用 llm.encode()，由 residual decoder 返回 Top-B 个完整四层 SID。

注意：
    - 这不是传统逐 token beam_search。
    - residual_sid_beam_size 从模型 config.json 中读取。
    - CType 与 SID_BEGIN 必须各自是 tokenizer 中的单 token。
"""

import os

os.environ.setdefault(
    "HF_ENDPOINT",
    "https://hf-mirror.com",
)
os.environ.setdefault(
    "VLLM_USE_V1",
    "0",
)
os.environ.setdefault(
    "VLLM_PLUGINS",
    "openonerec_residual_sid_v085",
)

import argparse
import json
import re
import time
import traceback
from multiprocessing import Process, Queue, set_start_method
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer


MID_LAYER_RES = {
    "a": re.compile(r"<mid_a_\d+>"),
    "b": re.compile(r"<mid_b_\d+>"),
    "c": re.compile(r"<mid_c_\d+>"),
}
MID_ANY_RE = re.compile(r"<mid_[abc]_\d+>")
CTYPE_RE = re.compile(r"<(?:\|)?ctype_\d+(?:\|)?>")
LS_RE = re.compile(r"<ls_[ab]_\d+>")

SID_BEGIN = "<|sid_begin|>"
SID_END = "<|sid_end|>"

SID_LAYER_RES = {
    "a": re.compile(r"<s_a_\d+>"),
    "b": re.compile(r"<s_b_\d+>"),
    "c": re.compile(r"<s_c_\d+>"),
    "d": re.compile(r"<s_d_\d+>"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate OpenOneRec-Res with MID-once + "
            "target-CType prefix inference."
        )
    )

    parser.add_argument(
        "--model-path",
        required=True,
    )
    parser.add_argument(
        "--data-path",
        required=True,
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
    )
    parser.add_argument(
        "--no-strict-validation",
        action="store_true",
    )

    return parser.parse_args()


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        if content.get("type") == "text":
            return str(content.get("text", ""))
        return ""

    if isinstance(content, list):
        chunks: List[str] = []

        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif (
                isinstance(item, dict)
                and item.get("type") == "text"
            ):
                chunks.append(
                    str(item.get("text", ""))
                )

        return "".join(chunks)

    raise ValueError(
        f"Unsupported content type: {type(content)}"
    )


def load_messages(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)

    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(
            "messages must contain input and final assistant target"
        )

    return value


def convert_messages(
    messages: Sequence[Dict[str, Any]],
) -> List[Dict[str, str]]:
    return [
        {
            "role": message["role"],
            "content": content_to_text(
                message.get("content", "")
            ),
        }
        for message in messages
    ]


def unique_match(
    text: str,
    pattern: re.Pattern,
    description: str,
) -> str:
    values = list(
        dict.fromkeys(
            pattern.findall(text)
        )
    )

    if len(values) != 1:
        raise ValueError(
            f"Expected exactly one {description}, got {values}; "
            f"text={text!r}"
        )

    return values[0]


def load_residual_config(
    model_path: str,
) -> Tuple[List[int], List[int], int, int]:
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    layer_starts = list(
        config.residual_sid_layer_starts
    )
    layer_sizes = list(
        config.residual_sid_layer_sizes
    )
    sid_begin_token_id = int(
        config.residual_sid_begin_token_id
    )
    beam_size = int(
        getattr(
            config,
            "residual_sid_beam_size",
            0,
        )
    )

    if (
        len(layer_starts) != 4
        or len(layer_sizes) != 4
    ):
        raise ValueError(
            "Expected exactly four SID layers: "
            f"starts={layer_starts}, sizes={layer_sizes}"
        )

    if beam_size <= 0:
        raise ValueError(
            "residual_sid_beam_size is missing "
            "from model config.json"
        )

    return (
        layer_starts,
        layer_sizes,
        sid_begin_token_id,
        beam_size,
    )


def extract_target_ctype(
    answer_text: str,
) -> str:
    ctype = unique_match(
        answer_text,
        CTYPE_RE,
        "target CType",
    )

    expected_prefix = (
        ctype
        + SID_BEGIN
    )

    if not answer_text.startswith(
        expected_prefix
    ):
        raise ValueError(
            "Assistant target must start with "
            "target CType + SID_BEGIN"
        )

    return ctype


def extract_target_sid(
    answer_text: str,
    tokenizer,
    layer_starts: Sequence[int],
    layer_sizes: Sequence[int],
) -> List[int]:
    answer_ids = tokenizer.encode(
        answer_text,
        add_special_tokens=False,
    )

    target: List[int] = []

    for layer_index, (start, size) in enumerate(
        zip(layer_starts, layer_sizes)
    ):
        matched = [
            token_id
            for token_id in answer_ids
            if start <= token_id < start + size
        ]

        if len(matched) != 1:
            raise ValueError(
                f"Cannot uniquely extract SID layer {layer_index}; "
                f"matched={matched}, answer={answer_text!r}"
            )

        target.append(matched[0])

    return target


def validate_sample(
    input_messages: Sequence[Dict[str, Any]],
    answer_text: str,
    row_context: str,
) -> None:
    input_text = "".join(
        content_to_text(
            message.get("content", "")
        )
        for message in input_messages
    )

    combined_text = (
        input_text
        + answer_text
    )

    if LS_RE.search(combined_text):
        raise ValueError(
            f"{row_context}: LS token remains"
        )

    # 整个 input 侧每层 MID 恰好出现一次。
    for layer in ("a", "b", "c"):
        matches = MID_LAYER_RES[layer].findall(
            input_text
        )

        if len(matches) != 1:
            raise ValueError(
                f"{row_context}: input side must contain exactly "
                f"one MID-{layer}, got {matches}"
            )

    # Assistant 不再重复 MID。
    if MID_ANY_RE.search(answer_text):
        raise ValueError(
            f"{row_context}: assistant target must not contain MID"
        )

    extract_target_ctype(answer_text)

    if (
        answer_text.count(SID_BEGIN) != 1
        or answer_text.count(SID_END) != 1
    ):
        raise ValueError(
            f"{row_context}: assistant must contain exactly one SID block"
        )

    for layer in ("a", "b", "c", "d"):
        unique_match(
            answer_text,
            SID_LAYER_RES[layer],
            f"target SID layer {layer}",
        )


def decode_pooling_output(
    output,
    beam_size: int,
    num_sid_layers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pooler output:
        beam_size × [sid_a, sid_b, sid_c, sid_d, cumulative_score]
    """
    data = output.outputs.data

    if hasattr(data, "cpu"):
        data = data.cpu().numpy()
    else:
        data = np.asarray(data)

    width = num_sid_layers + 1

    if data.ndim == 1:
        expected_numel = beam_size * width

        if data.size != expected_numel:
            raise ValueError(
                f"Unexpected pooling numel={data.size}, "
                f"expected={expected_numel}"
            )

        data = data.reshape(
            beam_size,
            width,
        )

    elif data.ndim == 2:
        expected_shape = (
            beam_size,
            width,
        )

        if data.shape != expected_shape:
            raise ValueError(
                f"Unexpected pooling shape={data.shape}, "
                f"expected={expected_shape}"
            )

    else:
        raise ValueError(
            f"Unexpected pooling ndim={data.ndim}, "
            f"shape={data.shape}"
        )

    candidate_ids = np.rint(
        data[:, :num_sid_layers]
    ).astype(np.int64)

    scores = data[:, num_sid_layers]

    return candidate_ids, scores


def run_batch(
    llm,
    prompt_list: List[Dict[str, Any]],
    target_list: List[List[int]],
    beam_size: int,
) -> Tuple[int, int]:
    outputs = llm.encode(
        prompt_list,
        use_tqdm=False,
    )

    if len(outputs) != len(target_list):
        raise RuntimeError(
            f"outputs={len(outputs)}, targets={len(target_list)}"
        )

    hit = 0

    for output, target in zip(
        outputs,
        target_list,
    ):
        candidate_ids, _ = decode_pooling_output(
            output=output,
            beam_size=beam_size,
            num_sid_layers=4,
        )

        target_array = np.asarray(
            target,
            dtype=np.int64,
        )

        matched = np.all(
            candidate_ids
            == target_array[None, :],
            axis=1,
        )

        hit += int(
            matched.any()
        )

    return hit, len(outputs)


def evaluate_worker(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    result_queue: Queue,
) -> None:
    try:
        os.environ[
            "CUDA_VISIBLE_DEVICES"
        ] = str(rank)

        from vllm import LLM

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )

        (
            layer_starts,
            layer_sizes,
            sid_begin_token_id,
            beam_size,
        ) = load_residual_config(
            args.model_path
        )

        print(
            f"[Rank {rank}] "
            f"beam_size={beam_size}, "
            f"SID starts={layer_starts}"
        )

        llm = LLM(
            model=args.model_path,
            task="embed",
            tensor_parallel_size=1,
            dtype=args.dtype,
            gpu_memory_utilization=(
                args.gpu_memory_utilization
            ),
            max_model_len=args.max_model_len,
            trust_remote_code=True,
        )

        dataframe = pd.read_parquet(
            args.data_path
        )

        dataframe = dataframe.iloc[
            rank::world_size
        ].reset_index(drop=True)

        prompt_list: List[Dict[str, Any]] = []
        target_list: List[List[int]] = []

        hit = 0
        count = 0
        started_at = time.perf_counter()

        for row_index, row in tqdm(
            dataframe.iterrows(),
            total=len(dataframe),
            desc=f"GPU-{rank}",
        ):
            messages = load_messages(
                row["messages"]
            )

            input_messages = messages[:-1]
            target_message = messages[-1]

            row_context = (
                f"rank={rank}, row={row_index}, "
                f"uuid={row.get('uuid', 'unknown')}"
            )

            if (
                str(
                    target_message.get(
                        "role",
                        "",
                    )
                ).lower()
                != "assistant"
            ):
                raise ValueError(
                    f"{row_context}: final message is not assistant"
                )

            answer_text = content_to_text(
                target_message.get(
                    "content",
                    "",
                )
            )

            if not args.no_strict_validation:
                validate_sample(
                    input_messages,
                    answer_text,
                    row_context,
                )

            # user prompt 内已经含 MID 一次，以及历史 CType+SID。
            chat_prompt_ids = tokenizer.apply_chat_template(
                convert_messages(
                    input_messages
                ),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            target_ctype = extract_target_ctype(
                answer_text
            )

            # MID 已在 user 上下文中，不再追加。
            # 已知 prefix 只有：target CType + SID_BEGIN。
            prefix_text = (
                target_ctype
                + SID_BEGIN
            )

            prefix_ids = tokenizer.encode(
                prefix_text,
                add_special_tokens=False,
            )

            if len(prefix_ids) != 2:
                raise ValueError(
                    f"{row_context}: target prefix tokenized into "
                    f"{len(prefix_ids)} tokens, expected 2. "
                    "CType and SID_BEGIN must each be one token."
                )

            if (
                prefix_ids[-1]
                != sid_begin_token_id
            ):
                raise ValueError(
                    f"{row_context}: prefix does not end with "
                    "the configured SID_BEGIN token"
                )

            prompt_token_ids = (
                list(chat_prompt_ids)
                + list(prefix_ids)
            )

            prompt_list.append(
                {
                    "prompt_token_ids": (
                        prompt_token_ids
                    ),
                }
            )

            target_list.append(
                extract_target_sid(
                    answer_text=answer_text,
                    tokenizer=tokenizer,
                    layer_starts=layer_starts,
                    layer_sizes=layer_sizes,
                )
            )

            if (
                len(prompt_list)
                >= args.batch_size
            ):
                (
                    batch_hit,
                    batch_count,
                ) = run_batch(
                    llm=llm,
                    prompt_list=prompt_list,
                    target_list=target_list,
                    beam_size=beam_size,
                )

                hit += batch_hit
                count += batch_count

                prompt_list.clear()
                target_list.clear()

        if prompt_list:
            (
                batch_hit,
                batch_count,
            ) = run_batch(
                llm=llm,
                prompt_list=prompt_list,
                target_list=target_list,
                beam_size=beam_size,
            )

            hit += batch_hit
            count += batch_count

        elapsed = (
            time.perf_counter()
            - started_at
        )

        result_queue.put(
            {
                "rank": rank,
                "hit": hit,
                "count": count,
                "elapsed": elapsed,
                "beam_size": beam_size,
            }
        )

        print(
            f"[Rank {rank}] "
            f"Recall@{beam_size}="
            f"{hit / max(count, 1):.6f}, "
            f"Time={elapsed:.2f}s"
        )

    except Exception as exc:
        trace = traceback.format_exc()
        print(trace)

        result_queue.put(
            {
                "rank": rank,
                "error": repr(exc),
                "traceback": trace,
            }
        )


def main() -> None:
    args = parse_args()

    if args.num_gpus <= 0:
        raise ValueError(
            "--num-gpus must be positive"
        )

    set_start_method(
        "spawn",
        force=True,
    )

    result_queue = Queue()
    processes: List[Process] = []

    for rank in range(
        args.num_gpus
    ):
        process = Process(
            target=evaluate_worker,
            args=(
                rank,
                args.num_gpus,
                args,
                result_queue,
            ),
        )

        process.start()
        processes.append(process)

    total_hit = 0
    total_count = 0
    worker_times: List[float] = []
    beam_size = None

    for _ in range(
        args.num_gpus
    ):
        result = result_queue.get()

        if "error" in result:
            for process in processes:
                if process.is_alive():
                    process.terminate()

            raise RuntimeError(
                f"Rank {result['rank']} failed: "
                f"{result['error']}\n"
                f"{result.get('traceback', '')}"
            )

        total_hit += result["hit"]
        total_count += result["count"]
        worker_times.append(
            result["elapsed"]
        )
        beam_size = result["beam_size"]

    for process in processes:
        process.join()

        if process.exitcode != 0:
            raise RuntimeError(
                f"Worker pid={process.pid} "
                f"exited with code {process.exitcode}"
            )

    wall_time = (
        max(worker_times)
        if worker_times
        else 0.0
    )

    recall = (
        total_hit
        / max(total_count, 1)
    )

    throughput = (
        total_count / wall_time
        if wall_time > 0
        else 0.0
    )

    print("=" * 72)
    print(
        f"Recall@{beam_size} = "
        f"{total_hit}/{total_count} "
        f"= {recall:.6f}"
    )
    print(
        f"Wall Time = {wall_time:.2f}s"
    )
    print(
        f"Throughput = {throughput:.2f} samples/s"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
