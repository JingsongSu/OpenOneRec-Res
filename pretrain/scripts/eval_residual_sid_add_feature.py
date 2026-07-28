#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenOneRec-Res residual SID 评测脚本。

适配当前测试集格式：

User:
    指令文本
    <mid_a_*><mid_b_*><mid_c_*>
    <history_ctype_1><|sid_begin|><history_sid_1><|sid_end|>
    <history_ctype_2><|sid_begin|><history_sid_2><|sid_end|>
    ...

Assistant:
    <target_ctype><|sid_begin|><target_sid><|sid_end|>

实际推理流程：

    1. system + user 经过 Qwen3 chat template；
    2. MID 已经在 user prompt 中出现一次，不再重复追加；
    3. 从 assistant ground truth 中提取 target CType；
    4. 在 assistant generation header 后追加：
           target CType + <|sid_begin|>
    5. prompt 最后一个 token 必须是 <|sid_begin|>；
    6. 调用 llm.encode()；
    7. residual decoder 返回 beam_size 个完整四层 SID；
    8. 统计 Recall@beam_size。

注意：
    - 不是传统 llm.beam_search()；
    - Transformer 只执行一次 prompt prefill；
    - beam_size 从模型 config.json 的 residual_sid_beam_size 读取；
    - CType 和 SID_BEGIN 必须分别是 tokenizer 中的单 token；
    - 参数全部在文件顶部写死，不使用 argparse。
"""

import os

# 必须在 import vllm 之前设置。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault(
    "VLLM_PLUGINS",
    "openonerec_residual_sid_v085",
)

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


# ============================================================================
# 固定配置：运行前修改这里
# ============================================================================

MODEL_PATH = (
    "/home/jovyan/ceph-1/sujinsong/sujinsong/OpenOneRec-res/pretrain/model_output/sft_full_residual_4layer_vllm085_b100-21000step"
)

# 指向已经转换为以下格式的测试集：
#   user      = MID_once + history CType/SID
#   assistant = target CType + SID_BEGIN + target SID + SID_END
DATA_PATH = (
    "/home/jovyan/ceph-1/sujinsong/sujinsong/OpenOneRec-res/output/eval/sft_video_rec_add_feature.parquet"
)

NUM_GPUS = 8
BATCH_SIZE = 128

GPU_MEMORY_UTILIZATION = 0.90
MAX_MODEL_LEN = 32768
DTYPE = "bfloat16"

# True：严格检查测试数据格式。
# 建议保持 True，避免旧 LS 格式或错误 prefix 混入评测。
STRICT_DATA_VALIDATION = True


# ============================================================================
# Token 格式
# ============================================================================

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


# ============================================================================
# 数据解析
# ============================================================================

def content_to_text(content: Any) -> str:
    """
    兼容 OpenOneRec parquet 中常见的 content 格式：
        1. str
        2. {"type": "text", "text": "..."}
        3. [{"type": "text", "text": "..."}]
    """
    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        if content.get("type") == "text":
            return str(content.get("text", ""))
        return ""

    if isinstance(content, list):
        result: List[str] = []

        for item in content:
            if isinstance(item, str):
                result.append(item)
            elif (
                isinstance(item, dict)
                and item.get("type") == "text"
            ):
                result.append(
                    str(item.get("text", ""))
                )

        return "".join(result)

    raise ValueError(
        f"Unsupported content type: {type(content)}"
    )


def load_messages(value: Any) -> List[Dict[str, Any]]:
    """
    messages 既可能是 JSON 字符串，也可能已经是 list。
    """
    if isinstance(value, str):
        value = json.loads(value)

    if not isinstance(value, list):
        raise ValueError(
            f"messages must be list or JSON string, got {type(value)}"
        )

    if len(value) < 2:
        raise ValueError(
            "messages must contain input messages "
            "and one final assistant target"
        )

    return value


def convert_messages(
    messages: Sequence[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """
    转为 tokenizer.apply_chat_template 所需的简单 role/content 格式。
    """
    result: List[Dict[str, str]] = []

    for message in messages:
        if "role" not in message:
            raise ValueError(
                f"Message has no role: {message}"
            )

        result.append(
            {
                "role": str(message["role"]),
                "content": content_to_text(
                    message.get("content", "")
                ),
            }
        )

    return result


def unique_match(
    text: str,
    pattern: re.Pattern,
    description: str,
) -> str:
    """
    要求 text 中只出现一个唯一 token。
    """
    values = list(
        dict.fromkeys(
            pattern.findall(text)
        )
    )

    if len(values) != 1:
        raise ValueError(
            f"Expected exactly one {description}, "
            f"got {values}; text={text!r}"
        )

    return values[0]


# ============================================================================
# Residual 模型配置
# ============================================================================

def load_residual_config(
    model_path: str,
) -> Tuple[List[int], List[int], int, int]:
    """
    从导出的 config.json 中读取 residual SID 推理配置。
    """
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    required_fields = [
        "residual_sid_layer_starts",
        "residual_sid_layer_sizes",
        "residual_sid_begin_token_id",
    ]

    for field in required_fields:
        if not hasattr(config, field):
            raise ValueError(
                f"Missing {field} in model config.json"
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

    if any(int(size) <= 0 for size in layer_sizes):
        raise ValueError(
            f"Invalid SID layer sizes: {layer_sizes}"
        )

    if beam_size <= 0:
        raise ValueError(
            "residual_sid_beam_size is missing or invalid "
            "in exported config.json"
        )

    return (
        [int(value) for value in layer_starts],
        [int(value) for value in layer_sizes],
        sid_begin_token_id,
        beam_size,
    )


# ============================================================================
# Target prefix / SID 提取
# ============================================================================

def extract_target_ctype(
    answer_text: str,
) -> str:
    """
    Assistant 必须是：

        <target_ctype><|sid_begin|>
        <s_a><s_b><s_c><s_d><|sid_end|>

    返回 target CType 字符串。
    """
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
            "target CType + SID_BEGIN; "
            f"answer={answer_text!r}"
        )

    return ctype


def extract_target_sid(
    answer_text: str,
    tokenizer,
    layer_starts: Sequence[int],
    layer_sizes: Sequence[int],
) -> List[int]:
    """
    从 assistant ground truth 中提取四层 SID global token ID。
    """
    answer_ids = tokenizer.encode(
        answer_text,
        add_special_tokens=False,
    )

    target: List[int] = []

    for layer_index, (start, size) in enumerate(
        zip(layer_starts, layer_sizes)
    ):
        end = start + size

        matched = [
            token_id
            for token_id in answer_ids
            if start <= token_id < end
        ]

        if len(matched) != 1:
            raise ValueError(
                "Cannot uniquely extract SID layer "
                f"{layer_index} in [{start}, {end}); "
                f"answer={answer_text!r}; "
                f"answer_ids={answer_ids}; "
                f"matched={matched}"
            )

        target.append(
            int(matched[0])
        )

    return target


# ============================================================================
# 数据格式校验
# ============================================================================

def validate_sample(
    input_messages: Sequence[Dict[str, Any]],
    answer_text: str,
    row_context: str,
) -> None:
    """
    校验当前 MID-once 测试集协议。
    """
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

    # 新词表和新数据都不应再出现 LS。
    if LS_RE.search(combined_text):
        raise ValueError(
            f"{row_context}: LS token remains"
        )

    # 整个 system+user 输入侧，MID-A/B/C 各出现一次。
    for layer in ("a", "b", "c"):
        matches = MID_LAYER_RES[layer].findall(
            input_text
        )

        if len(matches) != 1:
            raise ValueError(
                f"{row_context}: input side must contain "
                f"exactly one MID-{layer}, got {matches}"
            )

    # Assistant 不重复 MID。
    if MID_ANY_RE.search(answer_text):
        raise ValueError(
            f"{row_context}: assistant target must not contain MID"
        )

    target_ctype = extract_target_ctype(
        answer_text
    )

    # Assistant 目标必须只有一个完整 SID block。
    if answer_text.count(SID_BEGIN) != 1:
        raise ValueError(
            f"{row_context}: assistant must contain "
            f"exactly one SID_BEGIN"
        )

    if answer_text.count(SID_END) != 1:
        raise ValueError(
            f"{row_context}: assistant must contain "
            f"exactly one SID_END"
        )

    sid_tokens: List[str] = []

    for layer in ("a", "b", "c", "d"):
        sid_tokens.append(
            unique_match(
                answer_text,
                SID_LAYER_RES[layer],
                f"target SID layer {layer}",
            )
        )

    expected_answer = (
        target_ctype
        + SID_BEGIN
        + "".join(sid_tokens)
        + SID_END
    )

    if answer_text != expected_answer:
        raise ValueError(
            f"{row_context}: assistant target is not exactly "
            "CType + SID_BEGIN + A/B/C/D + SID_END; "
            f"expected={expected_answer!r}; "
            f"actual={answer_text!r}"
        )


def validate_prefix_tokenization(
    tokenizer,
    target_ctype: str,
    sid_begin_token_id: int,
    row_context: str,
) -> List[int]:
    """
    当前已知 prefix 只有两个 token：

        target CType + SID_BEGIN
    """
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
            f"{len(prefix_ids)} tokens, expected 2; "
            f"prefix={prefix_text!r}; ids={prefix_ids}. "
            "CType and SID_BEGIN must each be one tokenizer token."
        )

    if prefix_ids[-1] != sid_begin_token_id:
        raise ValueError(
            f"{row_context}: prefix does not end with "
            f"configured SID_BEGIN token ID; "
            f"ids={prefix_ids}, "
            f"expected_last={sid_begin_token_id}"
        )

    return [
        int(token_id)
        for token_id in prefix_ids
    ]


# ============================================================================
# Pooler 输出解析
# ============================================================================

def decode_pooling_output(
    output,
    beam_size: int,
    num_sid_layers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    residual pooler 输出：

        beam_size × [
            sid_a,
            sid_b,
            sid_c,
            sid_d,
            cumulative_score,
        ]
    """
    data = output.outputs.data

    if hasattr(data, "cpu"):
        data = data.cpu().numpy()
    else:
        data = np.asarray(data)

    expected_width = (
        num_sid_layers + 1
    )

    if data.ndim == 1:
        expected_numel = (
            beam_size
            * expected_width
        )

        if data.size != expected_numel:
            raise ValueError(
                "Unexpected residual pooling output: "
                f"shape={data.shape}, "
                f"numel={data.size}, "
                f"expected_numel={expected_numel}"
            )

        data = data.reshape(
            beam_size,
            expected_width,
        )

    elif data.ndim == 2:
        expected_shape = (
            beam_size,
            expected_width,
        )

        if data.shape != expected_shape:
            raise ValueError(
                "Unexpected residual pooling output: "
                f"shape={data.shape}, "
                f"expected_shape={expected_shape}"
            )

    else:
        raise ValueError(
            "Unexpected residual pooling output: "
            f"ndim={data.ndim}, "
            f"shape={data.shape}"
        )

    # 前四列：global SID token IDs。
    candidate_ids = np.rint(
        data[:, :num_sid_layers]
    ).astype(np.int64)

    # 最后一列：累计分数。
    scores = np.asarray(
        data[:, num_sid_layers],
        dtype=np.float64,
    )

    return (
        candidate_ids,
        scores,
    )


# ============================================================================
# Batch 推理
# ============================================================================

def run_batch(
    llm,
    prompt_list: List[Dict[str, Any]],
    target_list: List[List[int]],
    beam_size: int,
) -> Tuple[int, int]:
    """
    residual 核心推理。

    Transformer 只执行 prompt prefill：
        llm.encode()

    不使用：
        llm.beam_search()
    """
    if len(prompt_list) != len(target_list):
        raise ValueError(
            f"prompt_list={len(prompt_list)}, "
            f"target_list={len(target_list)}"
        )

    outputs = llm.encode(
        prompt_list,
        use_tqdm=False,
    )

    if len(outputs) != len(target_list):
        raise RuntimeError(
            f"outputs={len(outputs)}, "
            f"targets={len(target_list)}"
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

    return (
        hit,
        len(outputs),
    )


# ============================================================================
# 单 GPU Worker
# ============================================================================

def evaluate_worker(
    rank: int,
    world_size: int,
    result_queue: Queue,
) -> None:
    try:
        # 必须在 import vllm / 创建 LLM 前设置。
        os.environ[
            "CUDA_VISIBLE_DEVICES"
        ] = str(rank)

        # 每个进程独占一张 GPU。
        from vllm import LLM

        print(
            f"[Rank {rank}] "
            f"CUDA_VISIBLE_DEVICES={rank}"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
        )

        (
            layer_starts,
            layer_sizes,
            sid_begin_token_id,
            beam_size,
        ) = load_residual_config(
            MODEL_PATH
        )

        print(
            f"[Rank {rank}] "
            f"beam_size={beam_size}"
        )

        print(
            f"[Rank {rank}] "
            f"SID starts={layer_starts}"
        )

        print(
            f"[Rank {rank}] "
            f"SID sizes={layer_sizes}"
        )

        # 额外确认 tokenizer 中 SID_BEGIN 与 config 一致。
        tokenizer_sid_begin_id = (
            tokenizer.convert_tokens_to_ids(
                SID_BEGIN
            )
        )

        if (
            tokenizer_sid_begin_id
            != sid_begin_token_id
        ):
            raise ValueError(
                "Tokenizer/config SID_BEGIN mismatch: "
                f"tokenizer={tokenizer_sid_begin_id}, "
                f"config={sid_begin_token_id}"
            )

        print(
            f"[Rank {rank}] "
            "Loading residual vLLM model..."
        )

        llm = LLM(
            model=MODEL_PATH,
            task="embed",
            tensor_parallel_size=1,
            dtype=DTYPE,
            gpu_memory_utilization=(
                GPU_MEMORY_UTILIZATION
            ),
            max_model_len=MAX_MODEL_LEN,
            trust_remote_code=True,
        )

        dataframe = pd.read_parquet(
            DATA_PATH
        )

        if "messages" not in dataframe.columns:
            raise ValueError(
                f"DATA_PATH has no messages column: {DATA_PATH}"
            )

        # 多 GPU 按行号轮转切分。
        dataframe = dataframe.iloc[
            rank::world_size
        ].reset_index(
            drop=True
        )

        print(
            f"[Rank {rank}] "
            f"Assigned {len(dataframe)} samples"
        )

        hit = 0
        count = 0

        prompt_list: List[
            Dict[str, Any]
        ] = []

        target_list: List[
            List[int]
        ] = []

        start_time = time.perf_counter()

        for row_index, row in tqdm(
            dataframe.iterrows(),
            total=len(dataframe),
            desc=f"GPU-{rank}",
        ):
            row_context = (
                f"rank={rank}, "
                f"row={row_index}, "
                f"uuid={row.get('uuid', 'unknown')}"
            )

            messages_all = load_messages(
                row["messages"]
            )

            input_messages = messages_all[:-1]
            target_message = messages_all[-1]

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
                    f"{row_context}: final message "
                    "is not assistant"
                )

            answer_text = content_to_text(
                target_message.get(
                    "content",
                    "",
                )
            )

            if STRICT_DATA_VALIDATION:
                validate_sample(
                    input_messages=input_messages,
                    answer_text=answer_text,
                    row_context=row_context,
                )

            # system + user 经过 chat template。
            # 此时 user 中已经包含：
            #   MID_once + 历史 CType/SID。
            chat_prompt_ids = (
                tokenizer.apply_chat_template(
                    convert_messages(
                        input_messages
                    ),
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

            chat_prompt_ids = list(
                chat_prompt_ids
            )

            target_ctype = extract_target_ctype(
                answer_text
            )

            # 当前已知 assistant prefix：
            #   target CType + SID_BEGIN
            #
            # MID 已经在 user prompt 中，不再追加。
            prefix_ids = validate_prefix_tokenization(
                tokenizer=tokenizer,
                target_ctype=target_ctype,
                sid_begin_token_id=(
                    sid_begin_token_id
                ),
                row_context=row_context,
            )

            prompt_token_ids = (
                chat_prompt_ids
                + prefix_ids
            )

            if (
                not prompt_token_ids
                or prompt_token_ids[-1]
                != sid_begin_token_id
            ):
                raise ValueError(
                    f"{row_context}: final prompt "
                    "does not end with SID_BEGIN"
                )

            # vLLM 0.8.5 TokensPrompt。
            prompt_list.append(
                {
                    "prompt_token_ids":
                        prompt_token_ids
                }
            )

            target_sid = extract_target_sid(
                answer_text=answer_text,
                tokenizer=tokenizer,
                layer_starts=layer_starts,
                layer_sizes=layer_sizes,
            )

            target_list.append(
                target_sid
            )

            if len(prompt_list) >= BATCH_SIZE:
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

        # Flush 最后一批。
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
            - start_time
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

        ratio = (
            hit / count
            if count > 0
            else 0.0
        )

        print(
            f"[Rank {rank}] "
            f"Hit={hit}, "
            f"Count={count}, "
            f"Recall@{beam_size}={ratio:.6f}, "
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


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    if NUM_GPUS <= 0:
        raise ValueError(
            f"NUM_GPUS must be positive, got {NUM_GPUS}"
        )

    if BATCH_SIZE <= 0:
        raise ValueError(
            f"BATCH_SIZE must be positive, got {BATCH_SIZE}"
        )

    set_start_method(
        "spawn",
        force=True,
    )

    result_queue = Queue()
    processes: List[Process] = []

    for rank in range(NUM_GPUS):
        process = Process(
            target=evaluate_worker,
            args=(
                rank,
                NUM_GPUS,
                result_queue,
            ),
        )

        process.start()
        processes.append(
            process
        )

    total_hit = 0
    total_count = 0
    worker_times: List[float] = []
    beam_size = None

    try:
        for _ in range(NUM_GPUS):
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

            print(
                f"Receive rank {result['rank']} result: "
                f"{result['hit']}/{result['count']}"
            )

            total_hit += int(
                result["hit"]
            )

            total_count += int(
                result["count"]
            )

            worker_times.append(
                float(result["elapsed"])
            )

            beam_size = int(
                result["beam_size"]
            )

    finally:
        for process in processes:
            process.join()

    for process in processes:
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
        total_hit / total_count
        if total_count > 0
        else 0.0
    )

    throughput = (
        total_count / wall_time
        if wall_time > 0
        else 0.0
    )

    print("=" * 72)
    print("FINAL RESULT")
    print(
        f"Recall@{beam_size} = "
        f"{total_hit}/{total_count} "
        f"= {recall:.6f}"
    )
    print(
        f"Wall Time = "
        f"{wall_time:.2f} s"
    )
    print(
        f"Throughput = "
        f"{throughput:.2f} samples/s"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
