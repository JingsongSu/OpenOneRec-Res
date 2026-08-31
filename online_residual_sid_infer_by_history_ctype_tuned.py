#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenOneRec-Res residual SID 线上批量推理（safe-fast + time + latent 版）。

目标：
1. 不改变模型推理参数、beam、score 排序、CType 顺序、输出语义。
2. 完整保留当前 safe-fast 版吞吐优化：
   - CPU 输入预处理与 GPU encode 做多 batch 预取流水线；
   - residual 输出按整个 batch 向量化排序/转换；
   - GPU worker -> writer 的 SID payload 使用连续 NumPy ndarray，
     避免数百万 Python tuple 跨进程 pickle；
   - writer 保持 OUTPUT_WRITE_BUFFER_LINES=5_000_000 的按行 flush 逻辑；
   - 不启用 prefix caching：vLLM 0.8.5 V0 pooling runner 默认不初始化
     KV cache，直接开启 APC 并不是安全的 drop-in 优化。
3. 对齐 add_feature_time 输入协议：
   - all_parts_infer.parquet 中 messages 必须已经是
       MID + (ctype + SID + time) x history
   - 本脚本不重新计算 time，只校验并原样送入 chat template；
   - target/prediction prefix 仍然只追加
       <|ctype_x|><|sid_begin|>
     不追加 time，和训练 target 侧一致。
4. 对齐 branch-conditioned interleaved latent residual SID：
   - latent 不是文本 token，不向 messages 注入 <latent> / <ls_*>；
   - llm.encode() 得到 SID_BEGIN hidden 后，由 vLLM residual pooler 内部执行：
       A
       -> latent thought B -> formal residual B
       -> latent thought C -> formal residual C
       -> latent thought D -> formal residual D
   - 启动时硬校验 latent3 配置与 vLLM export version，避免误加载旧 residual 模型。

注意：
- 多 GPU producer 原本就是并发向 multiprocessing.Queue 写结果，因此不同
  GPU 之间的全局输出行顺序本身不保证跨运行 byte-identical。
- 对于同一个 (mid, ctype)，SID beam 排序及 adid 顺序保持原逻辑。
- time token 的生成/anchor 定义属于上游
  convert_all_parts_to_one_parquet_fast.py；这里绝不二次计算，避免
  train/serve 语义漂移。
"""

from __future__ import annotations

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("VLLM_PLUGINS", "openonerec_residual_sid_v085")

import json
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoConfig, AutoTokenizer


# ============================================================================
# 固定配置
# ============================================================================

PRETRAIN_ROOT = Path(
    "/home/jovyan/ceph-1/sujinsong/online/openonerec-res/pretrain"
)

CONVERTED_MODEL_PATH = Path(
    "/home/jovyan/ceph-1/sujinsong/online/openonerec-res/pretrain/"
    "model_output/sft_full_residual_add_feature_daily/step15000/"
    "global_step15000/converted"
)

RESIDUAL_CONFIG_PATH = Path(
    "/home/jovyan/ceph-1/sujinsong/online/openonerec-res/pretrain/"
    "model_output/sft_full_residual_add_feature_daily/residual_sid_config.json"
)

VLLM_MODEL_PATH = Path(
    "/home/jovyan/ceph-1/sujinsong/online/openonerec-res/pretrain/"
    "model_output/online_residual_vllm085_b100"
)

AUTO_PREPARE_VLLM_MODEL = True
FORCE_REEXPORT = True
EXPORT_BEAM_SIZE = 50

DATA_PATH = Path(
    "/home/jovyan/zhouyuhang-cloud1/sujingsong/online_infer/"
    "all_parts_infer.parquet"
)

ADID2SID_PARQUET = Path(
    "/home/jovyan/ceph-1/sujinsong/online/openonerec-res/"
    "raw_data/onerec_data/adid2sid.parquet"
)

CTYPE_IMAGE_SIZE_SEMID_TXT = Path(
    "/home/jovyan/zhouyuhang-cloud1/sujingsong/ctype_image_size_semid.txt"
)

OUTPUT_DIR = Path(
    "/home/jovyan/zhouyuhang-cloud1/sujingsong/online_infer/"
    "infer_adid_parts"
)

TARGET_CTYPES = ("3", "7", "2", "11", "12")

NUM_GPUS = 8
BATCH_SIZE = 10240
GPU_MEMORY_UTILIZATION = 0.95
MAX_MODEL_LEN = 13768
DTYPE = "bfloat16"
PARQUET_READ_BATCH_SIZE = 5000_000
PROGRESS_USERS = 5000_000
NUM_OUTPUT_PARTS = 200
OUTPUT_WRITE_BUFFER_LINES = 5000_000
OUTPUT_QUEUE_MAXSIZE = 64
PARQUET_USE_THREADS = False

# 保持你当前最新版参数不变。
INPUT_PREFETCH_BATCHES = 4


# ============================================================================
# Time / latent 协议常量
# ============================================================================

TIME_MIN_BUCKET = 0
TIME_MAX_BUCKET = 336

EXPECTED_LATENT_MODE = "branch_conditioned_interleaved"
EXPECTED_LATENT_STEPS = 3
EXPECTED_LATENT_TRANSITIONS = 3
EXPECTED_LATENT_CONDITIONING = "hard_previous_sid"
EXPECTED_LATENT_UPDATE = "thought_then_formal_residual"
MIN_RESIDUAL_VLLM_EXPORT_VERSION = 4
EXPECTED_VLLM_ARCHITECTURE = "Qwen3ForResidualSIDPoolingV085"


# ============================================================================
# Token / 数据解析
# ============================================================================

SID_BEGIN = "<|sid_begin|>"
SID_END = "<|sid_end|>"

CTYPE_RE = re.compile(r"<\|ctype_(\d+)\|>")
TIME_RE = re.compile(r"<\|time_(-?\d+)\|>")
LS_RE = re.compile(r"<ls_[ab]_\d+>")

# add_feature_time 的单个历史 item 必须严格是：
# <|ctype_x|><|sid_begin|><s_a_*><s_b_*><s_c_*><s_d_*><|sid_end|><|time_t|>
#
# 这里不要求整个 user 文本只有 token，因为前面还有自然语言 prompt 和 MID token；
# 只校验每个出现的 history ctype 都被完整的 SID + time 结构覆盖。
HISTORY_ITEM_RE = re.compile(
    r"<\|ctype_(\d+)\|>"
    r"<\|sid_begin\|>"
    r"<s_a_\d+>"
    r"<s_b_\d+>"
    r"<s_c_\d+>"
    r"<s_d_\d+>"
    r"<\|sid_end\|>"
    r"<\|time_(-?\d+)\|>"
)


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text", "")) if content.get("type") == "text" else ""
    if isinstance(content, list):
        result = []
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
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError("online messages must contain at least system + user")
    if any(str(x.get("role", "")).lower() == "assistant" for x in value):
        raise ValueError("online messages unexpectedly contain assistant")
    return value


def convert_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {
            "role": str(message["role"]),
            "content": content_to_text(message.get("content", "")),
        }
        for message in messages
    ]


def extract_mid(metadata_value: Any) -> str:
    metadata = json.loads(metadata_value) if isinstance(metadata_value, str) else metadata_value
    if not isinstance(metadata, dict):
        raise ValueError(f"Unsupported metadata: {type(metadata)}")
    mid = str(metadata.get("mid", "")).strip()
    if not mid:
        raise ValueError("metadata has empty mid")
    return mid


def has_nonzero_history_ctype(messages: Sequence[Dict[str, Any]]) -> bool:
    """
    保留原来的过滤语义：
      - history 没有 CType，或全部 CType=0 -> 不推理；
      - 只要存在非 0 CType -> 推理。

    同时增加 time+latent 版线上协议校验：
      1. 禁止旧 literal latent <ls_a_*> / <ls_b_*>；
      2. 每个 history item 必须严格为 ctype + SID + time；
      3. time bucket 必须落在 0..336；
      4. 不允许多出来的孤立 SID_BEGIN / SID_END / time token。

    latent 本身不在文本里；真正 latent reasoning 由 vLLM pooler 完成。
    """
    text = "".join(
        content_to_text(x.get("content", ""))
        for x in messages
    )

    if LS_RE.search(text):
        raise ValueError(
            "Old literal latent token <ls_*> remains in online input. "
            "The current latent path is hidden-state branch-conditioned "
            "interleaved reasoning and must not inject LS tokens."
        )

    ctypes = CTYPE_RE.findall(text)
    times = TIME_RE.findall(text)
    history_items = list(HISTORY_ITEM_RE.finditer(text))

    sid_begin_count = text.count(SID_BEGIN)
    sid_end_count = text.count(SID_END)

    if not ctypes:
        if times or sid_begin_count or sid_end_count:
            raise ValueError(
                "Online input contains SID/time tokens but no CType token; "
                "expected MID + (ctype + SID + time) x history."
            )
        return False

    if len(history_items) != len(ctypes):
        raise ValueError(
            "Online history is not add_feature_time format: "
            f"ctype_count={len(ctypes)}, "
            f"complete_history_item_count={len(history_items)}. "
            "Every CType must be immediately followed by "
            "SID_BEGIN + 4 SID layer tokens + SID_END + time."
        )

    if len(times) != len(history_items):
        raise ValueError(
            "Online history time-token count mismatch: "
            f"time_count={len(times)}, "
            f"history_item_count={len(history_items)}."
        )

    if sid_begin_count != len(history_items) or sid_end_count != len(history_items):
        raise ValueError(
            "Online history SID boundary count mismatch: "
            f"sid_begin={sid_begin_count}, sid_end={sid_end_count}, "
            f"history_items={len(history_items)}."
        )

    matched_ctypes: List[str] = []
    for item_index, match in enumerate(history_items):
        ctype = str(match.group(1))
        time_bucket = int(match.group(2))

        if not (TIME_MIN_BUCKET <= time_bucket <= TIME_MAX_BUCKET):
            raise ValueError(
                f"History time bucket outside [{TIME_MIN_BUCKET}, "
                f"{TIME_MAX_BUCKET}]: item={item_index}, "
                f"ctype={ctype}, time={time_bucket}"
            )

        matched_ctypes.append(ctype)

    # HISTORY_ITEM_RE 与 CTYPE_RE 都按文本顺序扫描，必须逐项一致。
    if matched_ctypes != ctypes:
        raise ValueError(
            "CType order mismatch while validating online history: "
            f"all_ctypes={ctypes[:16]}, matched_ctypes={matched_ctypes[:16]}"
        )

    return any(
        int(ctype) != 0
        for ctype in ctypes
    )


def validate_time_tokenizer(tokenizer) -> None:
    """
    确保 <|time_0|> ... <|time_336|> 共 337 个 token 在最终线上 tokenizer
    中全部存在，并且每个都恰好编码成 1 个 token。

    这是启动时一次性校验，不改变任何推理参数和 prompt 语义。
    """
    unk_token_id = getattr(tokenizer, "unk_token_id", None)

    for bucket in range(
        TIME_MIN_BUCKET,
        TIME_MAX_BUCKET + 1,
    ):
        token = f"<|time_{bucket}|>"
        token_id = tokenizer.convert_tokens_to_ids(token)

        if token_id is None:
            raise ValueError(f"Tokenizer is missing time token: {token}")

        token_id = int(token_id)

        if unk_token_id is not None and token_id == int(unk_token_id):
            raise ValueError(
                f"Time token resolves to unk_token_id: {token}"
            )

        encoded = tokenizer.encode(
            token,
            add_special_tokens=False,
        )

        if len(encoded) != 1 or int(encoded[0]) != token_id:
            raise ValueError(
                f"{token!r} must be exactly one tokenizer token, "
                f"convert_id={token_id}, encoded={encoded}"
            )


# ============================================================================
# latent / residual 配置校验
# ============================================================================

def validate_latent_residual_config(config: Any) -> None:
    """
    硬校验当前线上模型一定是 branch-conditioned interleaved latent3
    的 vLLM 0.8.5 residual export。

    这不会改变模型行为；只防止 VLLM_MODEL_PATH 指向旧 residual export 时
    静默跑成“没有 latent 的模型”。
    """
    architectures = list(
        getattr(
            config,
            "architectures",
            [],
        )
        or []
    )
    if architectures != [EXPECTED_VLLM_ARCHITECTURE]:
        raise ValueError(
            "Unexpected vLLM architecture: "
            f"{architectures!r}, "
            f"expected={[EXPECTED_VLLM_ARCHITECTURE]!r}"
        )

    export_version = int(
        getattr(
            config,
            "residual_sid_vllm_export_version",
            0,
        )
    )
    if export_version < MIN_RESIDUAL_VLLM_EXPORT_VERSION:
        raise ValueError(
            "Residual vLLM export is too old for branch-conditioned "
            "interleaved latent reasoning: "
            f"version={export_version}, "
            f"required>={MIN_RESIDUAL_VLLM_EXPORT_VERSION}"
        )

    target_version = str(
        getattr(
            config,
            "residual_sid_vllm_target_version",
            "",
        )
    )
    if target_version and target_version != "0.8.5":
        raise ValueError(
            "Unexpected residual SID vLLM target version: "
            f"{target_version!r}, expected '0.8.5'"
        )

    if not bool(
        getattr(
            config,
            "latent_reasoning_enabled",
            False,
        )
    ):
        raise ValueError(
            "latent_reasoning_enabled is not True; "
            "refusing to run a non-latent residual model."
        )

    latent_mode = str(
        getattr(
            config,
            "latent_reasoning_mode",
            "",
        )
    )
    if latent_mode != EXPECTED_LATENT_MODE:
        raise ValueError(
            "latent_reasoning_mode mismatch: "
            f"{latent_mode!r} != {EXPECTED_LATENT_MODE!r}"
        )

    latent_steps = int(
        getattr(
            config,
            "latent_reasoning_num_steps",
            0,
        )
    )
    if latent_steps != EXPECTED_LATENT_STEPS:
        raise ValueError(
            "latent_reasoning_num_steps mismatch: "
            f"{latent_steps} != {EXPECTED_LATENT_STEPS}"
        )

    latent_transitions = int(
        getattr(
            config,
            "latent_reasoning_num_transitions",
            0,
        )
    )
    if latent_transitions != EXPECTED_LATENT_TRANSITIONS:
        raise ValueError(
            "latent_reasoning_num_transitions mismatch: "
            f"{latent_transitions} != {EXPECTED_LATENT_TRANSITIONS}"
        )

    latent_conditioning = str(
        getattr(
            config,
            "latent_reasoning_conditioning",
            "",
        )
    )
    if latent_conditioning != EXPECTED_LATENT_CONDITIONING:
        raise ValueError(
            "latent_reasoning_conditioning mismatch: "
            f"{latent_conditioning!r} "
            f"!= {EXPECTED_LATENT_CONDITIONING!r}"
        )

    latent_update = str(
        getattr(
            config,
            "latent_reasoning_update",
            "",
        )
    )
    if latent_update != EXPECTED_LATENT_UPDATE:
        raise ValueError(
            "latent_reasoning_update mismatch: "
            f"{latent_update!r} != {EXPECTED_LATENT_UPDATE!r}"
        )

    # 当前 output decoder 固定按 [sid0, sid1, sid2, sid3, score] 解码。
    output_stride = int(
        getattr(
            config,
            "residual_sid_output_stride",
            5,
        )
    )
    if output_stride != 5:
        raise ValueError(
            "residual_sid_output_stride mismatch: "
            f"{output_stride} != 5"
        )


# ============================================================================
# converted -> vLLM085 residual export
# ============================================================================

def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_export_ready() -> bool:
    config_path = VLLM_MODEL_PATH / "config.json"
    custom_path = VLLM_MODEL_PATH / "model-residual-sid-vllm085.safetensors"
    summary_path = VLLM_MODEL_PATH / "residual_sid_vllm085_export.json"

    if not (
        config_path.is_file()
        and custom_path.is_file()
        and summary_path.is_file()
    ):
        return False

    try:
        config = read_json(config_path)
        summary = read_json(summary_path)

        return (
            config.get("architectures")
            == [EXPECTED_VLLM_ARCHITECTURE]
            and int(config.get("residual_sid_beam_size", 0))
            == EXPORT_BEAM_SIZE
            and int(config.get("residual_sid_vllm_export_version", 0))
            >= MIN_RESIDUAL_VLLM_EXPORT_VERSION
            and bool(config.get("latent_reasoning_enabled", False))
            and config.get("latent_reasoning_mode")
            == EXPECTED_LATENT_MODE
            and int(config.get("latent_reasoning_num_steps", 0))
            == EXPECTED_LATENT_STEPS
            and int(config.get("latent_reasoning_num_transitions", 0))
            == EXPECTED_LATENT_TRANSITIONS
            and config.get("latent_reasoning_conditioning")
            == EXPECTED_LATENT_CONDITIONING
            and config.get("latent_reasoning_update")
            == EXPECTED_LATENT_UPDATE
            and str(summary.get("source_model_dir", ""))
            == str(CONVERTED_MODEL_PATH.resolve())
        )
    except Exception:
        return False


def prepare_vllm_model() -> Path:
    if not AUTO_PREPARE_VLLM_MODEL:
        if not VLLM_MODEL_PATH.is_dir():
            raise FileNotFoundError(VLLM_MODEL_PATH)
        return VLLM_MODEL_PATH

    if not CONVERTED_MODEL_PATH.is_dir():
        raise FileNotFoundError(
            f"请先把 CONVERTED_MODEL_PATH 改成实际路径: {CONVERTED_MODEL_PATH}"
        )
    if not RESIDUAL_CONFIG_PATH.is_file():
        raise FileNotFoundError(RESIDUAL_CONFIG_PATH)

    patch_script = PRETRAIN_ROOT / "tools/model_converter/patch_residual_sid_hf_config.py"
    export_script = PRETRAIN_ROOT / "tools/model_converter/export_residual_sid_vllm085.py"

    if not patch_script.is_file() or not export_script.is_file():
        raise FileNotFoundError("residual model converter scripts not found")

    if not FORCE_REEXPORT and is_export_ready():
        print(f"Reuse existing vLLM model: {VLLM_MODEL_PATH}")
        return VLLM_MODEL_PATH

    print("Patching converted HF residual + latent config...")
    subprocess.run(
        [
            sys.executable,
            str(patch_script),
            "--hf_model_dir",
            str(CONVERTED_MODEL_PATH),
            "--residual_config",
            str(RESIDUAL_CONFIG_PATH),
        ],
        cwd=str(PRETRAIN_ROOT),
        check=True,
    )

    print("Exporting vLLM 0.8.5 latent residual model...")
    cmd = [
        sys.executable,
        str(export_script),
        "--source_model_dir",
        str(CONVERTED_MODEL_PATH),
        "--output_model_dir",
        str(VLLM_MODEL_PATH),
        "--beam_size",
        str(EXPORT_BEAM_SIZE),
    ]

    if FORCE_REEXPORT or VLLM_MODEL_PATH.exists():
        cmd.append("--overwrite")

    subprocess.run(
        cmd,
        cwd=str(PRETRAIN_ROOT),
        check=True,
    )

    if not is_export_ready():
        raise RuntimeError(
            "vLLM latent residual export validation failed"
        )

    return VLLM_MODEL_PATH


def load_residual_config(
    model_path: str,
) -> Tuple[List[int], List[int], int, int]:
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    validate_latent_residual_config(config)

    for name in (
        "residual_sid_layer_starts",
        "residual_sid_layer_sizes",
        "residual_sid_begin_token_id",
    ):
        if not hasattr(config, name):
            raise ValueError(f"Missing {name} in config")

    starts = [
        int(x)
        for x in config.residual_sid_layer_starts
    ]
    sizes = [
        int(x)
        for x in config.residual_sid_layer_sizes
    ]
    sid_begin_id = int(
        config.residual_sid_begin_token_id
    )
    beam_size = int(
        getattr(
            config,
            "residual_sid_beam_size",
            0,
        )
    )

    if len(starts) != 4 or len(sizes) != 4:
        raise ValueError(
            f"Expected 4 SID layers: starts={starts}, sizes={sizes}"
        )

    if beam_size != EXPORT_BEAM_SIZE:
        raise ValueError(
            f"beam size mismatch: config={beam_size}, "
            f"expected={EXPORT_BEAM_SIZE}"
        )

    return (
        starts,
        sizes,
        sid_begin_id,
        beam_size,
    )


def build_prefix_ids(
    tokenizer,
    sid_begin_id: int,
) -> Dict[str, List[int]]:
    result = {}

    for ctype in TARGET_CTYPES:
        text = f"<|ctype_{ctype}|>{SID_BEGIN}"
        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        if len(ids) != 2:
            raise ValueError(
                f"{text!r} must tokenize to 2 tokens, got {ids}"
            )

        if int(ids[-1]) != sid_begin_id:
            raise ValueError(
                f"{text!r} does not end with SID_BEGIN id"
            )

        result[ctype] = [
            int(x)
            for x in ids
        ]

    return result


# ============================================================================
# residual output -> local SID
# ============================================================================

def _pooling_output_matrix(
    output,
    beam_size: int,
) -> np.ndarray:
    """
    将单个 vLLM pooling output 标准化为 [beam_size, 5] NumPy 矩阵。
    数学逻辑与旧 decode_pooling_output 完全一致，只把 batch 级排序放到后面。
    """
    data = output.outputs.data

    if hasattr(data, "cpu"):
        data = data.cpu().numpy()
    else:
        data = np.asarray(data)

    width = 5

    if data.ndim == 1:
        if data.size != beam_size * width:
            raise ValueError(
                f"Unexpected pooler output: "
                f"shape={data.shape}, size={data.size}"
            )
        data = data.reshape(
            beam_size,
            width,
        )

    elif data.ndim == 2:
        if data.shape != (
            beam_size,
            width,
        ):
            raise ValueError(
                f"Unexpected pooler output: shape={data.shape}, "
                f"expected={(beam_size, width)}"
            )

    else:
        raise ValueError(
            f"Unexpected pooler ndim={data.ndim}, "
            f"shape={data.shape}"
        )

    return data


def decode_pooling_outputs_batch(
    outputs,
    beam_size: int,
    starts_np: np.ndarray,
    sizes_np: np.ndarray,
) -> np.ndarray:
    """
    旧逻辑：
      每个 output:
        score stable-desc sort
        rint global SID
        每条 beam Python 循环做 global->local

    新逻辑：
      将整个 llm.encode() 的输出堆成 [N, beam, 5]，
      用 NumPy 沿 beam 维做完全相同的 stable-desc sort，
      然后一次性完成 rint 和 global->local。

    返回：
      contiguous ndarray [N, beam, 4]，优先 int32；
      若 layer size 超过 int32 则自动使用 int64。

    注意：
      - cumulative_score 仍先转 float64；
      - 仍使用 np.argsort(-scores, kind="stable")；
      - SID 仍使用 np.rint；
      所以候选次序/取整逻辑与当前 safe-fast 版本保持一致。
    """
    if not outputs:
        dtype = (
            np.int32
            if int(
                np.max(
                    sizes_np,
                    initial=0,
                )
            )
            <= np.iinfo(np.int32).max
            else np.int64
        )

        return np.empty(
            (
                0,
                beam_size,
                4,
            ),
            dtype=dtype,
        )

    matrices = [
        _pooling_output_matrix(
            output,
            beam_size,
        )
        for output in outputs
    ]

    data = np.stack(
        matrices,
        axis=0,
    )

    scores = data[:, :, 4].astype(
        np.float64,
        copy=False,
    )

    if not np.all(
        np.isfinite(scores)
    ):
        bad = np.argwhere(
            ~np.isfinite(scores)
        )
        first = tuple(
            int(x)
            for x in bad[0]
        )

        raise ValueError(
            "Residual beam scores contain NaN or Inf: "
            f"first_bad_index={first}"
        )

    order = np.argsort(
        -scores,
        axis=1,
        kind="stable",
    )

    sorted_global = np.take_along_axis(
        data[:, :, :4],
        order[:, :, None],
        axis=1,
    )

    global_ids = np.rint(
        sorted_global
    ).astype(
        np.int64,
        copy=False,
    )

    local = (
        global_ids
        - starts_np.reshape(
            1,
            1,
            4,
        )
    )

    invalid = (
        (local < 0)
        | (
            local
            >= sizes_np.reshape(
                1,
                1,
                4,
            )
        )
    )

    if np.any(invalid):
        n, beam, layer = (
            int(x)
            for x in np.argwhere(invalid)[0]
        )

        token_id = int(
            global_ids[
                n,
                beam,
                layer,
            ]
        )
        start = int(
            starts_np[layer]
        )
        size = int(
            sizes_np[layer]
        )

        raise ValueError(
            f"SID token outside layer {layer}: "
            f"request={n}, beam={beam}, token_id={token_id}, "
            f"range=[{start}, {start + size})"
        )

    local_dtype = (
        np.int32
        if int(
            np.max(
                sizes_np,
                initial=0,
            )
        )
        <= np.iinfo(np.int32).max
        else np.int64
    )

    return np.ascontiguousarray(
        local.astype(
            local_dtype,
            copy=False,
        )
    )


# ============================================================================
# adid2sid reverse map
# ============================================================================

def normalize_key(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(
        value,
        (
            np.integer,
            int,
        ),
    ):
        return str(
            int(value)
        )

    if isinstance(
        value,
        (
            np.floating,
            float,
        ),
    ):
        if np.isnan(value):
            return ""
        if float(value).is_integer():
            return str(
                int(value)
            )

    return str(value).strip()


def parse_sid_value(
    value: Any,
) -> Tuple[int, int, int, int]:
    if isinstance(
        value,
        np.ndarray,
    ):
        parts = value.tolist()

    elif isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        parts = list(value)

    else:
        text = str(value).strip()

        if (
            text.startswith("[")
            and text.endswith("]")
        ):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None

            if isinstance(
                parsed,
                list,
            ):
                parts = parsed
            else:
                parts = [
                    x.strip()
                    for x in text.strip("[]").split(",")
                    if x.strip()
                ]

        else:
            parts = [
                x.strip()
                for x in text.split(",")
                if x.strip()
            ]

    if len(parts) != 4:
        raise ValueError(
            f"Expected 4-layer SID, got {value!r}"
        )

    return tuple(
        int(x)
        for x in parts
    )  # type: ignore[return-value]


def load_sid2adid() -> Dict[
    Tuple[int, int, int, int],
    str,
]:
    print(
        "[Writer] Loading adid2sid reverse map...",
        flush=True,
    )

    frame = pd.read_parquet(
        ADID2SID_PARQUET,
        columns=[
            "adid",
            "sid",
        ],
    )

    result: Dict[
        Tuple[int, int, int, int],
        str,
    ] = {}

    duplicate_same = 0

    for adid, sid in zip(
        frame["adid"],
        frame["sid"],
    ):
        adid_key = normalize_key(adid)

        if not adid_key:
            continue

        sid_key = parse_sid_value(sid)

        old = result.get(sid_key)

        if old is None:
            result[sid_key] = adid_key

        elif old == adid_key:
            duplicate_same += 1

        else:
            raise ValueError(
                f"SID maps to multiple adids: sid={sid_key}, "
                f"adid1={old}, adid2={adid_key}"
            )

    print(
        f"[Writer] sid2adid={len(result):,}, "
        f"duplicate_same={duplicate_same:,}",
        flush=True,
    )

    return result


# ============================================================================
# CType -> 输出 ctype 映射
# ============================================================================

def load_ctype_output_map() -> Dict[str, str]:
    print(
        f"[Writer] Loading ctype map: "
        f"{CTYPE_IMAGE_SIZE_SEMID_TXT}",
        flush=True,
    )

    result: Dict[
        str,
        str,
    ] = {}

    with CTYPE_IMAGE_SIZE_SEMID_TXT.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_no, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if (
                not line
                or line.startswith("#")
            ):
                continue

            cols = line.split()

            if len(cols) < 2:
                raise ValueError(
                    f"Bad ctype map line {line_no}: "
                    f"{line!r}"
                )

            raw_ctype = str(
                cols[0]
            ).strip()
            mapped_ctype = str(
                cols[1]
            ).strip()

            old = result.get(
                raw_ctype
            )

            if (
                old is not None
                and old != mapped_ctype
            ):
                raise ValueError(
                    f"ctype={raw_ctype} has multiple mappings: "
                    f"{old!r} vs {mapped_ctype!r}"
                )

            result[
                raw_ctype
            ] = mapped_ctype

    missing = [
        ctype
        for ctype in TARGET_CTYPES
        if ctype not in result
    ]

    if missing:
        raise ValueError(
            f"TARGET_CTYPES missing in ctype map: {missing}"
        )

    print(
        "[Writer] ctype map loaded: "
        + ", ".join(
            f"{ctype}->{result[ctype]}"
            for ctype in TARGET_CTYPES
        ),
        flush=True,
    )

    return result


# ============================================================================
# 输出 part writer
# ============================================================================

class PartWriter:
    def __init__(self) -> None:
        self.part_index = 0
        self.current_lines = 0
        self.total_lines = 0
        self.buffer: List[str] = []

        self.temp_path = (
            OUTPUT_DIR
            / ".all_output.tmp"
        )

        if self.temp_path.exists():
            self.temp_path.unlink()

        self.handle = self.temp_path.open(
            "w",
            encoding="utf-8",
            buffering=8 * 1024 * 1024,
        )

        print(
            f"[Writer] Temporary output: "
            f"{self.temp_path}",
            flush=True,
        )

    def _flush(self) -> None:
        if not self.buffer:
            return

        if self.handle is None:
            raise RuntimeError(
                "Temporary output file is not open"
            )

        self.handle.write(
            "".join(
                self.buffer
            )
        )
        self.buffer.clear()

    def append(
        self,
        mid_ctype: str,
        adids: Sequence[str],
    ) -> None:
        if self.handle is None:
            raise RuntimeError(
                "Temporary output file is closed"
            )

        line = (
            "\t".join(
                [
                    mid_ctype,
                    *adids,
                ]
            )
            + "\n"
        )

        self.buffer.append(
            line
        )

        self.current_lines += 1
        self.total_lines += 1

        if (
            len(self.buffer)
            >= OUTPUT_WRITE_BUFFER_LINES
        ):
            self._flush()

    def close(self) -> None:
        if self.handle is not None:
            self._flush()
            self.handle.close()
            self.handle = None

    def finalize(self) -> None:
        self.close()
        self.part_index = 0

        base_lines, remainder = divmod(
            self.total_lines,
            NUM_OUTPUT_PARTS,
        )

        print(
            f"[Writer] Splitting {self.total_lines:,} lines "
            f"into {NUM_OUTPUT_PARTS} parts: "
            f"base={base_lines:,}, remainder={remainder:,}",
            flush=True,
        )

        with self.temp_path.open(
            "r",
            encoding="utf-8",
            buffering=8 * 1024 * 1024,
        ) as source:
            for part_index in range(
                NUM_OUTPUT_PARTS
            ):
                lines_in_part = (
                    base_lines
                    + (
                        1
                        if part_index < remainder
                        else 0
                    )
                )

                part_path = (
                    OUTPUT_DIR
                    / f"part-{part_index:04d}"
                )

                with part_path.open(
                    "w",
                    encoding="utf-8",
                    buffering=8 * 1024 * 1024,
                ) as target:
                    target.writelines(
                        islice(
                            source,
                            lines_in_part,
                        )
                    )

                self.part_index += 1

                print(
                    f"[Writer] Saved {part_path}: "
                    f"{lines_in_part:,} lines",
                    flush=True,
                )

            extra_line = source.readline()

            if extra_line:
                raise RuntimeError(
                    "Temporary output contains more lines "
                    "than writer.total_lines"
                )

        self.temp_path.unlink()

        print(
            f"[Writer] Split completed: "
            f"{self.part_index} parts, "
            f"{self.total_lines:,} total lines",
            flush=True,
        )


# ============================================================================
# CPU writer / SID reverse map process
# ============================================================================

def output_writer_worker(
    output_queue,
    status_queue,
    num_gpu_workers: int,
) -> None:
    try:
        sid2adid = load_sid2adid()
        ctype_output_map = load_ctype_output_map()
        writer = PartWriter()

        done_workers = 0
        sid_hit = 0
        sid_miss = 0
        prefixes = 0

        try:
            while True:
                message = output_queue.get()
                kind = message[0]

                if kind == "batch":
                    # safe-fast payload:
                    # (
                    #   "batch",
                    #   mids: List[str],
                    #   ctypes: List[str],
                    #   local_sid_batch: np.ndarray [N, beam, 4]
                    # )
                    mids = message[1]
                    ctypes = message[2]
                    sid_batch = message[3]

                    if not isinstance(
                        sid_batch,
                        np.ndarray,
                    ):
                        raise TypeError(
                            "sid_batch must be ndarray, "
                            f"got {type(sid_batch)}"
                        )

                    if (
                        sid_batch.ndim != 3
                        or sid_batch.shape[2] != 4
                    ):
                        raise ValueError(
                            f"Unexpected sid_batch "
                            f"shape={sid_batch.shape}"
                        )

                    if (
                        len(mids) != len(ctypes)
                        or len(mids)
                        != sid_batch.shape[0]
                    ):
                        raise ValueError(
                            "writer payload length mismatch: "
                            f"mids={len(mids)}, "
                            f"ctypes={len(ctypes)}, "
                            f"sid_batch={sid_batch.shape}"
                        )

                    for row_index, (
                        mid,
                        ctype,
                    ) in enumerate(
                        zip(
                            mids,
                            ctypes,
                        )
                    ):
                        prefixes += 1
                        adids: List[str] = []

                        # 与当前 safe-fast 版完全相同：
                        # 按 latent residual pooler 返回的 beam 顺序
                        # 逐个 reverse-map。
                        for sid_row in sid_batch[
                            row_index
                        ]:
                            sid_key = (
                                int(sid_row[0]),
                                int(sid_row[1]),
                                int(sid_row[2]),
                                int(sid_row[3]),
                            )

                            adid = sid2adid.get(
                                sid_key
                            )

                            if adid is None:
                                sid_miss += 1
                                continue

                            sid_hit += 1
                            adids.append(
                                adid
                            )

                        if not adids:
                            continue

                        mapped_ctype = (
                            ctype_output_map.get(
                                ctype
                            )
                        )

                        if mapped_ctype is None:
                            raise KeyError(
                                f"ctype={ctype!r} missing in "
                                f"{CTYPE_IMAGE_SIZE_SEMID_TXT}"
                            )

                        mid_ctype = (
                            f"{mid}_{mapped_ctype}"
                        )

                        writer.append(
                            mid_ctype=mid_ctype,
                            adids=adids,
                        )

                elif kind == "worker_done":
                    done_workers += 1

                    print(
                        f"[Writer] GPU done "
                        f"{done_workers}/"
                        f"{num_gpu_workers}",
                        flush=True,
                    )

                    if (
                        done_workers
                        >= num_gpu_workers
                    ):
                        break

                elif kind == "abort":
                    raise RuntimeError(
                        f"Abort requested by GPU "
                        f"{message[1]}"
                    )

                else:
                    raise ValueError(
                        f"Unknown writer message: "
                        f"{kind}"
                    )

        finally:
            writer.close()

        writer.finalize()

        status_queue.put(
            {
                "kind": "writer",
                "ok": True,
                "output_lines": writer.total_lines,
                "num_parts": writer.part_index,
                "sid_hit": sid_hit,
                "sid_miss": sid_miss,
                "prefixes": prefixes,
            }
        )

    except Exception as exc:
        trace = traceback.format_exc()
        print(
            trace,
            flush=True,
        )

        status_queue.put(
            {
                "kind": "writer",
                "ok": False,
                "error": repr(exc),
                "traceback": trace,
            }
        )


# ============================================================================
# GPU batch
# ============================================================================

def run_residual_batch(
    llm,
    prompt_list: List[
        Dict[str, Any]
    ],
    prompt_meta: List[
        Tuple[str, str]
    ],
    beam_size: int,
    starts_np: np.ndarray,
    sizes_np: np.ndarray,
    output_queue,
) -> Tuple[
    int,
    float,
    float,
    float,
]:
    if len(prompt_list) != len(prompt_meta):
        raise ValueError(
            "prompt/meta length mismatch"
        )

    t0 = time.perf_counter()

    # 关键：
    # 这里仍然只调用 llm.encode()。
    #
    # time:
    #   已经在 prompt_token_ids 中，参与 Transformer prefill。
    #
    # latent:
    #   不作为 token 出现在 prompt 中。
    #   Qwen3ForResidualSIDPoolingV085 的 pooler 会在 SID_BEGIN hidden 上
    #   内部执行 branch-conditioned interleaved latent -> formal residual。
    outputs = llm.encode(
        prompt_list,
        use_tqdm=False,
    )

    t1 = time.perf_counter()

    if len(outputs) != len(prompt_meta):
        raise RuntimeError(
            "vLLM output/meta length mismatch"
        )

    local_sid_batch = decode_pooling_outputs_batch(
        outputs=outputs,
        beam_size=beam_size,
        starts_np=starts_np,
        sizes_np=sizes_np,
    )

    mids = [
        meta[0]
        for meta in prompt_meta
    ]
    ctypes = [
        meta[1]
        for meta in prompt_meta
    ]

    t2 = time.perf_counter()

    output_queue.put(
        (
            "batch",
            mids,
            ctypes,
            local_sid_batch,
        )
    )

    t3 = time.perf_counter()

    return (
        len(outputs),
        t1 - t0,
        t2 - t1,
        t3 - t2,
    )


# ============================================================================
# 输入 producer：与 GPU encode 做预取流水线
# ============================================================================

def _put_prefetch_item(
    ready_queue: "queue.Queue",
    item,
    stop_event: threading.Event,
) -> bool:
    while not stop_event.is_set():
        try:
            ready_queue.put(
                item,
                timeout=0.5,
            )
            return True
        except queue.Full:
            continue

    return False


def prompt_batch_producer(
    rank: int,
    world_size: int,
    tokenizer,
    prefix_ids_map: Dict[
        str,
        List[int],
    ],
    sid_begin_id: int,
    ready_queue: "queue.Queue",
    stop_event: threading.Event,
    stats: Counter,
    start_time: float,
) -> None:
    """
    负责：
      parquet -> JSON/messages -> time 协议校验
      -> chat template -> prompt batches

    prompt 构造顺序与当前 safe-fast 版完全一致：
      user1 按 TARGET_CTYPES 顺序，
      user2 按 TARGET_CTYPES 顺序，
      ...

    新结构：
      messages 中 history 已经包含
      MID + (ctype + SID + time) x history。

    prediction prefix 仍然只追加：
      <|ctype_target|><|sid_begin|>

    不追加 time，也不追加任何 latent literal token。
    """
    try:
        parquet_file = pq.ParquetFile(
            DATA_PATH
        )

        missing = {
            "messages",
            "metadata",
        } - set(
            parquet_file.schema.names
        )

        if missing:
            raise ValueError(
                "Input parquet missing columns: "
                f"{sorted(missing)}"
            )

        num_row_groups = (
            parquet_file.num_row_groups
        )

        rg_start = (
            num_row_groups
            * rank
            // world_size
        )

        rg_end = (
            num_row_groups
            * (rank + 1)
            // world_size
        )

        assigned_row_groups = list(
            range(
                rg_start,
                rg_end,
            )
        )

        print(
            f"[Rank {rank}] row_groups="
            f"{len(assigned_row_groups)}/"
            f"{parquet_file.num_row_groups}",
            flush=True,
        )

        prompt_list: List[
            Dict[str, Any]
        ] = []
        prompt_meta: List[
            Tuple[str, str]
        ] = []

        for row_group_index in assigned_row_groups:
            if stop_event.is_set():
                return

            batches = parquet_file.iter_batches(
                batch_size=(
                    PARQUET_READ_BATCH_SIZE
                ),
                row_groups=[
                    row_group_index
                ],
                columns=[
                    "messages",
                    "metadata",
                ],
                use_threads=(
                    PARQUET_USE_THREADS
                ),
            )

            for batch in batches:
                if stop_event.is_set():
                    return

                data = batch.to_pydict()

                for (
                    messages_value,
                    metadata_value,
                ) in zip(
                    data["messages"],
                    data["metadata"],
                ):
                    if stop_event.is_set():
                        return

                    stats["users_seen"] += 1

                    messages = load_messages(
                        messages_value
                    )

                    mid = extract_mid(
                        metadata_value
                    )

                    if not has_nonzero_history_ctype(
                        messages
                    ):
                        stats[
                            "users_all_zero_or_no_ctype"
                        ] += 1
                        continue

                    stats["users_inferred"] += 1
                    stats["prefixes_created"] += len(
                        TARGET_CTYPES
                    )

                    # time token 已经包含在 messages 中，
                    # 因此会在这里直接进入 prompt_token_ids。
                    base_ids = tokenizer.apply_chat_template(
                        convert_messages(
                            messages
                        ),
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )

                    base_ids = list(
                        base_ids
                    )

                    for ctype in TARGET_CTYPES:
                        # 与训练 target 侧一致：
                        # 只加 target CType + SID_BEGIN。
                        # 不加 target time。
                        # latent 由 pooler 内部完成，也不加 literal token。
                        prompt_token_ids = (
                            base_ids
                            + prefix_ids_map[
                                ctype
                            ]
                        )

                        if (
                            not prompt_token_ids
                            or prompt_token_ids[
                                -1
                            ]
                            != sid_begin_id
                        ):
                            raise ValueError(
                                "Prompt does not end "
                                "with SID_BEGIN"
                            )

                        prompt_list.append(
                            {
                                "prompt_token_ids":
                                    prompt_token_ids
                            }
                        )

                        prompt_meta.append(
                            (
                                mid,
                                ctype,
                            )
                        )

                        if (
                            len(prompt_list)
                            >= BATCH_SIZE
                        ):
                            if not _put_prefetch_item(
                                ready_queue,
                                (
                                    "batch",
                                    prompt_list,
                                    prompt_meta,
                                ),
                                stop_event,
                            ):
                                return

                            # queued batch 由 GPU thread 持有，
                            # producer 必须换新 list。
                            prompt_list = []
                            prompt_meta = []

                    if (
                        stats["users_seen"]
                        % PROGRESS_USERS
                        == 0
                    ):
                        elapsed = (
                            time.perf_counter()
                            - start_time
                        )

                        print(
                            f"[Rank {rank}] "
                            f"users="
                            f"{stats['users_seen']:,}; "
                            f"inferred="
                            f"{stats['users_inferred']:,}; "
                            f"skip="
                            f"{stats['users_all_zero_or_no_ctype']:,}; "
                            f"prefixes="
                            f"{stats['prefixes_created']:,}; "
                            f"time={elapsed:.1f}s",
                            flush=True,
                        )

        if prompt_list:
            if not _put_prefetch_item(
                ready_queue,
                (
                    "batch",
                    prompt_list,
                    prompt_meta,
                ),
                stop_event,
            ):
                return

        _put_prefetch_item(
            ready_queue,
            (
                "done",
            ),
            stop_event,
        )

    except Exception as exc:
        trace = traceback.format_exc()

        _put_prefetch_item(
            ready_queue,
            (
                "error",
                repr(exc),
                trace,
            ),
            stop_event,
        )


# ============================================================================
# 单 GPU worker
# ============================================================================

def inference_worker(
    rank: int,
    world_size: int,
    model_path: str,
    output_queue,
    status_queue,
) -> None:
    stop_event: (
        threading.Event
        | None
    ) = None

    producer_thread: (
        threading.Thread
        | None
    ) = None

    try:
        os.environ[
            "CUDA_VISIBLE_DEVICES"
        ] = str(rank)

        from vllm import LLM

        print(
            f"[Rank {rank}] "
            f"CUDA_VISIBLE_DEVICES={rank}",
            flush=True,
        )

        tokenizer = (
            AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
        )

        # time vocab 只做启动时校验，不改变 tokenizer。
        validate_time_tokenizer(
            tokenizer
        )

        (
            starts,
            sizes,
            sid_begin_id,
            beam_size,
        ) = load_residual_config(
            model_path
        )

        starts_np = np.asarray(
            starts,
            dtype=np.int64,
        )

        sizes_np = np.asarray(
            sizes,
            dtype=np.int64,
        )

        tokenizer_sid_begin_id = (
            tokenizer.convert_tokens_to_ids(
                SID_BEGIN
            )
        )

        if (
            int(tokenizer_sid_begin_id)
            != sid_begin_id
        ):
            raise ValueError(
                "Tokenizer/config SID_BEGIN mismatch: "
                f"tokenizer={tokenizer_sid_begin_id}, "
                f"config={sid_begin_id}"
            )

        prefix_ids_map = build_prefix_ids(
            tokenizer,
            sid_begin_id,
        )

        print(
            f"[Rank {rank}] beam={beam_size}, "
            f"starts={starts}, sizes={sizes}, "
            f"time_tokens="
            f"{TIME_MIN_BUCKET}..{TIME_MAX_BUCKET}, "
            f"latent={EXPECTED_LATENT_MODE}/"
            f"{EXPECTED_LATENT_STEPS}",
            flush=True,
        )

        # 保持你当前最新版 vLLM 参数完全不变。
        llm = LLM(
            model=model_path,
            task="embed",
            tensor_parallel_size=1,
            dtype=DTYPE,
            gpu_memory_utilization=(
                GPU_MEMORY_UTILIZATION
            ),
            max_model_len=MAX_MODEL_LEN,
            trust_remote_code=True,
        )

        stats: Counter = Counter()
        start_time = time.perf_counter()

        ready_queue: "queue.Queue" = (
            queue.Queue(
                maxsize=(
                    INPUT_PREFETCH_BATCHES
                )
            )
        )

        stop_event = (
            threading.Event()
        )

        producer_thread = (
            threading.Thread(
                target=(
                    prompt_batch_producer
                ),
                args=(
                    rank,
                    world_size,
                    tokenizer,
                    prefix_ids_map,
                    sid_begin_id,
                    ready_queue,
                    stop_event,
                    stats,
                    start_time,
                ),
                name=(
                    f"rank-{rank}-"
                    f"input-producer"
                ),
                daemon=True,
            )
        )

        producer_thread.start()

        batch_index = 0

        while True:
            item = ready_queue.get()
            kind = item[0]

            if kind == "done":
                break

            if kind == "error":
                raise RuntimeError(
                    "Input producer failed:\n"
                    f"{item[1]}\n"
                    f"{item[2]}"
                )

            if kind != "batch":
                raise ValueError(
                    "Unknown input producer "
                    f"message: {kind}"
                )

            prompt_list = item[1]
            prompt_meta = item[2]

            (
                processed,
                encode_seconds,
                post_seconds,
                queue_seconds,
            ) = run_residual_batch(
                llm=llm,
                prompt_list=prompt_list,
                prompt_meta=prompt_meta,
                beam_size=beam_size,
                starts_np=starts_np,
                sizes_np=sizes_np,
                output_queue=output_queue,
            )

            stats[
                "prefixes_processed"
            ] += processed
            stats[
                "encode_seconds"
            ] += encode_seconds
            stats[
                "post_seconds"
            ] += post_seconds
            stats[
                "output_queue_seconds"
            ] += queue_seconds
            stats[
                "gpu_batches"
            ] += 1

            batch_index += 1

        stop_event.set()
        producer_thread.join()

        elapsed = (
            time.perf_counter()
            - start_time
        )

        output_queue.put(
            (
                "worker_done",
                rank,
            )
        )

        status_queue.put(
            {
                "kind": "gpu",
                "rank": rank,
                "ok": True,
                "elapsed": elapsed,
                "beam_size": beam_size,
                "stats": dict(stats),
            }
        )

        print(
            f"[Rank {rank}] DONE "
            f"users="
            f"{stats['users_seen']:,}, "
            f"inferred="
            f"{stats['users_inferred']:,}, "
            f"prefixes="
            f"{stats['prefixes_processed']:,}, "
            f"encode="
            f"{stats['encode_seconds']:.2f}s, "
            f"post="
            f"{stats['post_seconds']:.2f}s, "
            f"queue="
            f"{stats['output_queue_seconds']:.2f}s, "
            f"time={elapsed:.2f}s",
            flush=True,
        )

    except Exception as exc:
        if stop_event is not None:
            stop_event.set()

        trace = traceback.format_exc()

        print(
            trace,
            flush=True,
        )

        try:
            output_queue.put(
                (
                    "abort",
                    rank,
                ),
                timeout=2,
            )
        except Exception:
            pass

        status_queue.put(
            {
                "kind": "gpu",
                "rank": rank,
                "ok": False,
                "error": repr(exc),
                "traceback": trace,
            }
        )


# ============================================================================
# Main
# ============================================================================

def check_paths() -> None:
    for path in [
        PRETRAIN_ROOT,
        DATA_PATH,
        ADID2SID_PARQUET,
        CTYPE_IMAGE_SIZE_SEMID_TXT,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def clean_output_parts() -> None:
    for path in OUTPUT_DIR.glob(
        "part-*"
    ):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(
                path
            )


def main() -> None:
    check_paths()

    if (
        NUM_GPUS <= 0
        or BATCH_SIZE <= 0
    ):
        raise ValueError(
            "NUM_GPUS and BATCH_SIZE "
            "must be positive"
        )

    if INPUT_PREFETCH_BATCHES <= 0:
        raise ValueError(
            "INPUT_PREFETCH_BATCHES "
            "must be positive"
        )

    model_path = prepare_vllm_model()

    print("")
    print("=" * 80)
    print(
        "ONLINE RESIDUAL SID INFERENCE "
        "(SAFE-FAST + TIME + LATENT)"
    )
    print("=" * 80)
    print(
        f"Converted model : "
        f"{CONVERTED_MODEL_PATH}"
    )
    print(
        f"vLLM model      : "
        f"{model_path}"
    )
    print(
        f"Input parquet   : "
        f"{DATA_PATH}"
    )
    print(
        f"Target CType    : "
        f"{TARGET_CTYPES}"
    )
    print(
        f"GPUs            : "
        f"{NUM_GPUS}"
    )
    print(
        f"Prefix batch    : "
        f"{BATCH_SIZE}"
    )
    print(
        f"Input prefetch  : "
        f"{INPUT_PREFETCH_BATCHES}"
    )
    print(
        f"Time buckets    : "
        f"{TIME_MIN_BUCKET}.."
        f"{TIME_MAX_BUCKET}"
    )
    print(
        f"Latent mode     : "
        f"{EXPECTED_LATENT_MODE}"
    )
    print(
        f"Latent steps    : "
        f"{EXPECTED_LATENT_STEPS}"
    )
    print(
        f"Output parts    : "
        f"{NUM_OUTPUT_PARTS}"
    )
    print(
        f"Output dir      : "
        f"{OUTPUT_DIR}"
    )
    print("=" * 80)

    clean_output_parts()

    import multiprocessing as mp

    ctx = mp.get_context(
        "spawn"
    )

    output_queue = ctx.Queue(
        maxsize=(
            OUTPUT_QUEUE_MAXSIZE
        )
    )

    status_queue = ctx.Queue()

    writer_process = ctx.Process(
        target=(
            output_writer_worker
        ),
        args=(
            output_queue,
            status_queue,
            NUM_GPUS,
        ),
        name="result-writer",
    )

    writer_process.start()

    gpu_processes = []

    for rank in range(
        NUM_GPUS
    ):
        process = ctx.Process(
            target=(
                inference_worker
            ),
            args=(
                rank,
                NUM_GPUS,
                str(model_path),
                output_queue,
                status_queue,
            ),
            name=f"gpu-{rank}",
        )

        process.start()
        gpu_processes.append(
            process
        )

    overall_start = (
        time.perf_counter()
    )

    gpu_results = []
    writer_result = None
    failed = None

    try:
        while (
            len(gpu_results)
            < NUM_GPUS
            or writer_result is None
        ):
            status = (
                status_queue.get()
            )

            if not status.get(
                "ok",
                False,
            ):
                failed = status
                break

            if status["kind"] == "gpu":
                gpu_results.append(
                    status
                )

                print(
                    "Receive GPU "
                    f"{status['rank']} result",
                    flush=True,
                )

            elif status["kind"] == "writer":
                writer_result = status

                print(
                    "Receive writer result",
                    flush=True,
                )

            else:
                raise ValueError(
                    f"Unknown status: "
                    f"{status}"
                )

        if failed is not None:
            for process in gpu_processes:
                if process.is_alive():
                    process.terminate()

            if writer_process.is_alive():
                writer_process.terminate()

            raise RuntimeError(
                f"{failed['kind']} failed: "
                f"{failed.get('error')}\n"
                f"{failed.get('traceback', '')}"
            )

    finally:
        for process in gpu_processes:
            process.join()

        writer_process.join()

    for process in gpu_processes:
        if process.exitcode != 0:
            raise RuntimeError(
                f"{process.name} "
                f"exitcode={process.exitcode}"
            )

    if writer_process.exitcode != 0:
        raise RuntimeError(
            f"writer exitcode="
            f"{writer_process.exitcode}"
        )

    if writer_result is None:
        raise RuntimeError(
            "Missing writer result"
        )

    total_stats: Counter = Counter()
    worker_times = []
    beam_size = None

    for result in gpu_results:
        total_stats.update(
            result["stats"]
        )

        worker_times.append(
            float(
                result["elapsed"]
            )
        )

        beam_size = int(
            result["beam_size"]
        )

    wall_time = (
        max(worker_times)
        if worker_times
        else 0.0
    )

    total_elapsed = (
        time.perf_counter()
        - overall_start
    )

    print("")
    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)
    print(
        f"Beam size            : "
        f"{beam_size}"
    )
    print(
        f"Users seen           : "
        f"{total_stats['users_seen']:,}"
    )
    print(
        f"Users inferred       : "
        f"{total_stats['users_inferred']:,}"
    )
    print(
        f"Users all-zero/no CType: "
        f"{total_stats['users_all_zero_or_no_ctype']:,}"
    )
    print(
        f"Prefix prompts       : "
        f"{total_stats['prefixes_processed']:,}"
    )
    print(
        f"SID reverse-map hit  : "
        f"{writer_result['sid_hit']:,}"
    )
    print(
        f"SID reverse-map miss : "
        f"{writer_result['sid_miss']:,}"
    )
    print(
        f"Output lines         : "
        f"{writer_result['output_lines']:,}"
    )
    print(
        f"Output parts         : "
        f"{writer_result['num_parts']:,}"
    )
    print(
        f"GPU wall time        : "
        f"{wall_time:.2f} s"
    )
    print(
        f"Total elapsed        : "
        f"{total_elapsed:.2f} s"
    )
    print(
        f"Sum encode time      : "
        f"{total_stats['encode_seconds']:.2f} s"
    )
    print(
        f"Sum post time        : "
        f"{total_stats['post_seconds']:.2f} s"
    )
    print(
        f"Sum output queue time: "
        f"{total_stats['output_queue_seconds']:.2f} s"
    )

    if wall_time > 0:
        print(
            f"User throughput      : "
            f"{total_stats['users_seen'] / wall_time:,.2f} "
            f"users/s"
        )
        print(
            f"Prefix throughput    : "
            f"{total_stats['prefixes_processed'] / wall_time:,.2f} "
            f"prefixes/s"
        )

    print(
        f"Output dir           : "
        f"{OUTPUT_DIR}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
