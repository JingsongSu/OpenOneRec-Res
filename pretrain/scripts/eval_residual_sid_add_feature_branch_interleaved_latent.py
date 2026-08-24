#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenOneRec-Res branch-conditioned interleaved latent residual SID evaluation.

Raw data and prompt format stay exactly in the original residual-SID form:

    <target_ctype><|sid_begin|>

No latent tokens are appended. A is decoded first from the raw SID_BEGIN
hidden. Before B/C/D, each surviving hard beam branch performs one internal
latent thought conditioned on its actual previous SID, followed immediately by
the formal residual transition for that layer.

Recall computation and beam output format are unchanged.
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("VLLM_PLUGINS", "openonerec_residual_sid_v085")

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
# Fixed config
# ============================================================================

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    (
        "/home/jovyan/ceph-1/sujinsong/sujinsong/OpenOneRec-latent/"
        "pretrain/model_output/"
        "sft_full_residual_add_feature_branch_interleaved_latent3_vllm085_b100-20000step"
    ),
)

DATA_PATH = os.environ.get(
    "DATA_PATH",
    (
        "/home/jovyan/ceph-1/sujinsong/sujinsong/OpenOneRec-latent/"
        "output/eval/sft_video_rec_add_feature.parquet"
    ),
)

NUM_GPUS = 8
BATCH_SIZE = 128
GPU_MEMORY_UTILIZATION = 0.90
MAX_MODEL_LEN = 32768
DTYPE = "bfloat16"
STRICT_DATA_VALIDATION = True


# ============================================================================
# Token format
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
# Data helpers
# ============================================================================

def content_to_text(content: Any) -> str:
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
            elif isinstance(item, dict) and item.get("type") == "text":
                result.append(str(item.get("text", "")))
        return "".join(result)
    raise ValueError(f"Unsupported content type: {type(content)}")


def load_messages(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError(
            f"messages must be list or JSON string, got {type(value)}"
        )
    if len(value) < 2:
        raise ValueError(
            "messages must contain input messages and one final assistant target"
        )
    return value


def convert_messages(
    messages: Sequence[Dict[str, Any]],
) -> List[Dict[str, str]]:
    result: List[Dict[str, str]] = []
    for message in messages:
        if "role" not in message:
            raise ValueError(f"Message has no role: {message}")
        result.append(
            {
                "role": str(message["role"]),
                "content": content_to_text(message.get("content", "")),
            }
        )
    return result


def unique_match(
    text: str,
    pattern: re.Pattern,
    description: str,
) -> str:
    values = list(dict.fromkeys(pattern.findall(text)))
    if len(values) != 1:
        raise ValueError(
            f"Expected exactly one {description}, "
            f"got {values}; text={text!r}"
        )
    return values[0]


# ============================================================================
# Model config
# ============================================================================

def load_residual_config(
    model_path: str,
) -> Tuple[List[int], List[int], int, int]:
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    required_fields = (
        "residual_sid_layer_starts",
        "residual_sid_layer_sizes",
        "residual_sid_begin_token_id",
    )
    for field in required_fields:
        if not hasattr(config, field):
            raise ValueError(f"Missing {field} in model config.json")

    starts = [int(x) for x in config.residual_sid_layer_starts]
    sizes = [int(x) for x in config.residual_sid_layer_sizes]
    sid_begin_id = int(config.residual_sid_begin_token_id)
    beam_size = int(getattr(config, "residual_sid_beam_size", 0))

    if len(starts) != 4 or len(sizes) != 4:
        raise ValueError(
            f"Expected four SID layers, got starts={starts}, sizes={sizes}"
        )
    if any(x <= 0 for x in sizes):
        raise ValueError(f"Invalid SID layer sizes: {sizes}")
    if beam_size <= 0:
        raise ValueError(
            "residual_sid_beam_size is missing or invalid in config.json"
        )
    return starts, sizes, sid_begin_id, beam_size


def validate_branch_interleaved_latent_config(
    model_path: str,
) -> None:
    """Fail early unless the exported config matches this experiment."""
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    enabled = bool(
        getattr(
            config,
            "latent_reasoning_enabled",
            False,
        )
    )
    mode = str(
        getattr(
            config,
            "latent_reasoning_mode",
            "",
        )
    )
    num_steps = int(
        getattr(
            config,
            "latent_reasoning_num_steps",
            0,
        )
    )
    num_transitions = int(
        getattr(
            config,
            "latent_reasoning_num_transitions",
            0,
        )
    )
    conditioning = str(
        getattr(config, "latent_reasoning_conditioning", "")
    )
    update = str(
        getattr(config, "latent_reasoning_update", "")
    )
    aux_weight = float(
        getattr(
            config,
            "latent_reasoning_loss_weight",
            0.0,
        )
    )

    if not enabled:
        raise ValueError(
            "latent_reasoning_enabled is missing/False in exported config.json"
        )
    if mode != "branch_conditioned_interleaved":
        raise ValueError(
            "Expected latent_reasoning_mode="
            "'branch_conditioned_interleaved', "
            f"got {mode!r}"
        )
    if num_steps != 3:
        raise ValueError(
            "latent_reasoning_num_steps must be 3 interleaved thoughts "
            "before B/C/D, "
            f"got {num_steps}"
        )
    if num_transitions != 3:
        raise ValueError(
            "latent_reasoning_num_transitions must be 3, "
            f"got {num_transitions}"
        )
    if conditioning != "hard_previous_sid":
        raise ValueError(
            "latent_reasoning_conditioning must be 'hard_previous_sid', "
            f"got {conditioning!r}"
        )
    if update != "thought_then_formal_residual":
        raise ValueError(
            "latent_reasoning_update must be 'thought_then_formal_residual', "
            f"got {update!r}"
        )
    if aux_weight != 0.0:
        raise ValueError(
            "This branch-conditioned experiment uses no auxiliary latent CE; "
            f"latent_reasoning_loss_weight={aux_weight}"
        )

    print(
        "Branch-conditioned interleaved latent config:",
        {
            "mode": mode,
            "steps": num_steps,
            "transitions": num_transitions,
            "dropout": float(
                getattr(
                    config,
                    "latent_reasoning_dropout",
                    0.0,
                )
            ),
            "conditioning": conditioning,
            "update": update,
            "aux_loss_weight": aux_weight,
        },
    )


# ============================================================================
# Target extraction and validation
# ============================================================================

def extract_target_ctype(answer_text: str) -> str:
    # Raw parquet remains: CType + SID_BEGIN + SID + SID_END.
    ctype = unique_match(answer_text, CTYPE_RE, "target CType")
    expected_prefix = ctype + SID_BEGIN
    if not answer_text.startswith(expected_prefix):
        raise ValueError(
            "Assistant target must start with target CType + SID_BEGIN; "
            f"answer={answer_text!r}"
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
                f"answer={answer_text!r}; answer_ids={answer_ids}; "
                f"matched={matched}"
            )
        target.append(int(matched[0]))

    return target


def validate_sample(
    input_messages: Sequence[Dict[str, Any]],
    answer_text: str,
    row_context: str,
) -> None:
    input_text = "".join(
        content_to_text(message.get("content", ""))
        for message in input_messages
    )
    combined_text = input_text + answer_text

    if LS_RE.search(combined_text):
        raise ValueError(f"{row_context}: LS token remains")


    for layer in ("a", "b", "c"):
        matches = MID_LAYER_RES[layer].findall(input_text)
        if len(matches) != 1:
            raise ValueError(
                f"{row_context}: input side must contain exactly one "
                f"MID-{layer}, got {matches}"
            )

    if MID_ANY_RE.search(answer_text):
        raise ValueError(
            f"{row_context}: assistant target must not contain MID"
        )

    target_ctype = extract_target_ctype(answer_text)

    if answer_text.count(SID_BEGIN) != 1:
        raise ValueError(
            f"{row_context}: assistant must contain exactly one SID_BEGIN"
        )
    if answer_text.count(SID_END) != 1:
        raise ValueError(
            f"{row_context}: assistant must contain exactly one SID_END"
        )

    sid_tokens = [
        unique_match(
            answer_text,
            SID_LAYER_RES[layer],
            f"target SID layer {layer}",
        )
        for layer in ("a", "b", "c", "d")
    ]

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
            f"expected={expected_answer!r}; actual={answer_text!r}"
        )


def validate_prefix_tokenization(
    tokenizer,
    target_ctype: str,
    sid_begin_token_id: int,
    row_context: str,
) -> List[int]:
    """Original residual-SID inference prefix: CType + SID_BEGIN."""
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

    if int(prefix_ids[-1]) != int(sid_begin_token_id):
        raise ValueError(
            f"{row_context}: prefix does not end with SID_BEGIN; "
            f"ids={prefix_ids}, expected_last={sid_begin_token_id}"
        )

    return [
        int(token_id)
        for token_id in prefix_ids
    ]


# ============================================================================
# Pooler output
# ============================================================================

def decode_pooling_output(
    output,
    beam_size: int,
    num_sid_layers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    data = output.outputs.data

    if hasattr(data, "cpu"):
        data = data.cpu().numpy()
    else:
        data = np.asarray(data)

    expected_width = num_sid_layers + 1

    if data.ndim == 1:
        expected_numel = beam_size * expected_width
        if data.size != expected_numel:
            raise ValueError(
                "Unexpected residual pooling output: "
                f"shape={data.shape}, numel={data.size}, "
                f"expected_numel={expected_numel}"
            )
        data = data.reshape(beam_size, expected_width)

    elif data.ndim == 2:
        expected_shape = (beam_size, expected_width)
        if data.shape != expected_shape:
            raise ValueError(
                "Unexpected residual pooling output: "
                f"shape={data.shape}, expected_shape={expected_shape}"
            )

    else:
        raise ValueError(
            "Unexpected residual pooling output: "
            f"ndim={data.ndim}, shape={data.shape}"
        )

    candidate_ids = np.rint(
        data[:, :num_sid_layers]
    ).astype(np.int64)

    scores = np.asarray(
        data[:, num_sid_layers],
        dtype=np.float64,
    )

    return candidate_ids, scores


# ============================================================================
# Batch inference
# ============================================================================

def run_batch(
    llm,
    prompt_list: List[Dict[str, Any]],
    target_list: List[List[int]],
    beam_size: int,
) -> Tuple[int, int]:
    if len(prompt_list) != len(target_list):
        raise ValueError(
            f"prompt_list={len(prompt_list)}, "
            f"target_list={len(target_list)}"
        )

    # Same as the original residual evaluator.
    # Transformer performs prompt prefill once; residual pooler decodes SID beam.
    outputs = llm.encode(
        prompt_list,
        use_tqdm=False,
    )

    if len(outputs) != len(target_list):
        raise RuntimeError(
            f"outputs={len(outputs)}, targets={len(target_list)}"
        )

    hit = 0

    for output, target in zip(outputs, target_list):
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
            candidate_ids == target_array[None, :],
            axis=1,
        )

        hit += int(matched.any())

    return hit, len(outputs)


# ============================================================================
# Single-GPU worker
# ============================================================================

def evaluate_worker(
    rank: int,
    world_size: int,
    result_queue: Queue,
) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

        # Import after CUDA_VISIBLE_DEVICES is fixed.
        from vllm import LLM

        print(
            f"[Rank {rank}] CUDA_VISIBLE_DEVICES={rank}"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
        )

        if tokenizer.chat_template is None:
            raise ValueError(
                "tokenizer.chat_template is None in exported model. "
                "The SFT/converted/exported tokenizer must preserve "
                "the Qwen3 chat template."
            )

        (
            layer_starts,
            layer_sizes,
            sid_begin_token_id,
            beam_size,
        ) = load_residual_config(
            MODEL_PATH
        )

        validate_branch_interleaved_latent_config(
            MODEL_PATH
        )

        print(
            f"[Rank {rank}] beam_size={beam_size}"
        )
        print(
            f"[Rank {rank}] SID starts={layer_starts}"
        )
        print(
            f"[Rank {rank}] SID sizes={layer_sizes}"
        )

        tokenizer_sid_begin_id = (
            tokenizer.convert_tokens_to_ids(
                SID_BEGIN
            )
        )

        if tokenizer_sid_begin_id != sid_begin_token_id:
            raise ValueError(
                "Tokenizer/config SID_BEGIN mismatch: "
                f"tokenizer={tokenizer_sid_begin_id}, "
                f"config={sid_begin_token_id}"
            )

        sid_begin_encoded = tokenizer.encode(
            SID_BEGIN,
            add_special_tokens=False,
        )

        if sid_begin_encoded != [sid_begin_token_id]:
            raise ValueError(
                "SID_BEGIN is not atomic: "
                f"encoded={sid_begin_encoded}, "
                f"expected={[sid_begin_token_id]}"
            )

        print(
            f"[Rank {rank}] Loading branch-conditioned interleaved latent residual vLLM model..."
        )

        llm = LLM(
            model=MODEL_PATH,
            task="embed",
            tensor_parallel_size=1,
            dtype=DTYPE,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
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

        dataframe = dataframe.iloc[
            rank::world_size
        ].reset_index(
            drop=True
        )

        print(
            f"[Rank {rank}] Assigned {len(dataframe)} samples"
        )

        hit = 0
        count = 0

        prompt_list: List[Dict[str, Any]] = []
        target_list: List[List[int]] = []

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
                    f"{row_context}: final message is not assistant"
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

            # Exactly the original chat-template path.
            # Historical SID remains untouched in input_messages.
            chat_prompt_ids = tokenizer.apply_chat_template(
                convert_messages(
                    input_messages
                ),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            chat_prompt_ids = list(
                chat_prompt_ids
            )

            target_ctype = extract_target_ctype(
                answer_text
            )

            # Prompt stays in the original residual-SID format:
            #   CType + SID_BEGIN
            #
            # The vLLM pooler performs the four-step soft SID A/B/C/D
            # lookahead internally on hidden(SID_BEGIN).
            prefix_ids = validate_prefix_tokenization(
                tokenizer=tokenizer,
                target_ctype=target_ctype,
                sid_begin_token_id=sid_begin_token_id,
                row_context=row_context,
            )

            prompt_token_ids = (
                chat_prompt_ids
                + prefix_ids
            )

            # Residual pooler still sees SID_BEGIN as the final prompt token.
            if (
                not prompt_token_ids
                or prompt_token_ids[-1] != sid_begin_token_id
            ):
                raise ValueError(
                    f"{row_context}: final prompt does not end with SID_BEGIN"
                )

            prompt_list.append(
                {
                    "prompt_token_ids":
                        prompt_token_ids
                }
            )

            # Ground truth remains the raw four-layer SID.
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

    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(
            f"MODEL_PATH does not exist: {MODEL_PATH}"
        )

    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(
            f"DATA_PATH does not exist: {DATA_PATH}"
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
        processes.append(process)

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
