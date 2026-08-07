#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenOneRec-Res residual SID 线上批量推理。

流程：
  big parquet(system+user)
    -> 检查历史序列中是否出现过非 0 CType
    -> 符合条件的用户固定分别追加：
         <|ctype_3|><|sid_begin|>
         <|ctype_7|><|sid_begin|>
         <|ctype_2|><|sid_begin|>
    -> 每个用户固定执行三次 llm.encode() residual beam
    -> global SID token IDs 转四层本地 SID code
    -> adid2sid.parquet 反查 adid
    -> part-*：mid_<mapped_ctype><TAB>adid1<TAB>...<TAB>adid100

历史序列 CType 全为 0（或没有 CType token）的用户直接跳过。
最终输出平均切分为 100 个 part，无表头。

模型输入是最新训练得到的 converted HF 模型。
可在启动时自动：
  patch_residual_sid_hf_config.py
  export_residual_sid_vllm085.py
然后加载 vLLM085 residual 导出目录。
"""

from __future__ import annotations

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("VLLM_PLUGINS", "openonerec_residual_sid_v085")

import json
import re
import shutil
import subprocess
import sys
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

# 改成你当前最新更新的 converted 路径。
CONVERTED_MODEL_PATH = Path(
    "/home/jovyan/ceph-1/sujinsong/online/openonerec-res/pretrain/model_output/sft_full_residual_add_feature_daily/step15000/global_step15000/converted"
)

RESIDUAL_CONFIG_PATH = Path(
    "/home/jovyan/ceph-1/sujinsong/online/openonerec-res/pretrain/model_output/sft_full_residual_add_feature_daily/residual_sid_config.json"
)

VLLM_MODEL_PATH = Path(
    "/home/jovyan/ceph-1/sujinsong/online/openonerec-res/pretrain/model_output/online_residual_vllm085_b100"
)

AUTO_PREPARE_VLLM_MODEL = True

# converted 仍会持续更新时保持 True。
# 模型固定后重复跑同一个模型可改 False。
# converted 更新后第一次跑设 True；导出完成后重复跑数据请设 False。
# 否则每次启动都会重新复制/导出整个大模型，看起来会像“卡住”。
FORCE_REEXPORT = True

EXPORT_BEAM_SIZE = 300

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

# 只要历史序列中出现过任意非 0 CType，
# 就固定按下面顺序分别推理三次。
TARGET_CTYPES = ("3", "7", "2")

NUM_GPUS = 8

# 展开 CType 后的 prefix prompt 数，不是用户数。
# prefix prompt batch。128 偏保守；先用 256。
# 如果显存稳定且 GPU 利用率仍低，可以再试 384 / 512。
BATCH_SIZE = 10240

GPU_MEMORY_UTILIZATION = 0.95

# 512 个历史 item 大约只有几千 token。
# 32768 会增加 vLLM 初始化/显存压力；线上先用 8192。
# 如果实际 prompt 超过 8192，再提高到 12288/16384。
MAX_MODEL_LEN = 13768
DTYPE = "bfloat16"

# 大 Parquet 减少 Python/Arrow batch 切换次数。
# 内存足够可继续提高到 100_000。
PARQUET_READ_BATCH_SIZE = 5000_000

# 只影响日志频率，不影响推理结果；调小便于确认程序没有卡死。
PROGRESS_USERS = 5000_000

# 最终固定平均切分成 100 个 part。
NUM_OUTPUT_PARTS = 100

# 降低频繁小写磁盘开销。
OUTPUT_WRITE_BUFFER_LINES = 5000_000

# 让 GPU 在 writer 短时跟不上时还能继续推几批。
# 不建议无限增大，因为每个 batch 含 beam SID，Python 对象比较大。
OUTPUT_QUEUE_MAXSIZE = 64

# 8 个独立 GPU 进程已经并行读 Parquet。
# 对 Ceph/NFS/网络盘，单进程内部再开 Arrow 多线程常会造成 I/O/CPU 争抢。
# 如果 DATA_PATH 在本地 NVMe，可改 True 测试。
PARQUET_USE_THREADS = False


# ============================================================================
# Token / 数据解析
# ============================================================================

SID_BEGIN = "<|sid_begin|>"
CTYPE_RE = re.compile(r"<\|ctype_(\d+)\|>")
LS_RE = re.compile(r"<ls_[ab]_\d+>")


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


def has_nonzero_history_ctype(
    messages: Sequence[Dict[str, Any]],
) -> bool:
    """
    只判断原始用户历史序列中是否出现过非 0 CType。

    - 任意一个 CType != 0：该用户固定推理 3、7、2。
    - CType 全为 0：跳过。
    - 没有任何 CType token：跳过。
    """
    text = "".join(
        content_to_text(x.get("content", ""))
        for x in messages
    )

    if LS_RE.search(text):
        raise ValueError("LS token remains in online input")

    history_ctypes = CTYPE_RE.findall(text)

    return any(
        int(ctype) != 0
        for ctype in history_ctypes
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
    if not (config_path.is_file() and custom_path.is_file() and summary_path.is_file()):
        return False
    try:
        config = read_json(config_path)
        summary = read_json(summary_path)
        return (
            config.get("architectures") == ["Qwen3ForResidualSIDPoolingV085"]
            and int(config.get("residual_sid_beam_size", 0)) == EXPORT_BEAM_SIZE
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

    print("Patching converted HF residual config...")
    subprocess.run(
        [
            sys.executable, str(patch_script),
            "--hf_model_dir", str(CONVERTED_MODEL_PATH),
            "--residual_config", str(RESIDUAL_CONFIG_PATH),
        ],
        cwd=str(PRETRAIN_ROOT),
        check=True,
    )

    print("Exporting vLLM 0.8.5 residual model...")
    cmd = [
        sys.executable, str(export_script),
        "--source_model_dir", str(CONVERTED_MODEL_PATH),
        "--output_model_dir", str(VLLM_MODEL_PATH),
        "--beam_size", str(EXPORT_BEAM_SIZE),
    ]
    if FORCE_REEXPORT or VLLM_MODEL_PATH.exists():
        cmd.append("--overwrite")

    subprocess.run(cmd, cwd=str(PRETRAIN_ROOT), check=True)

    if not is_export_ready():
        raise RuntimeError("vLLM residual export validation failed")
    return VLLM_MODEL_PATH


def load_residual_config(model_path: str) -> Tuple[List[int], List[int], int, int]:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    for name in (
        "residual_sid_layer_starts",
        "residual_sid_layer_sizes",
        "residual_sid_begin_token_id",
    ):
        if not hasattr(config, name):
            raise ValueError(f"Missing {name} in config")

    starts = [int(x) for x in config.residual_sid_layer_starts]
    sizes = [int(x) for x in config.residual_sid_layer_sizes]
    sid_begin_id = int(config.residual_sid_begin_token_id)
    beam_size = int(getattr(config, "residual_sid_beam_size", 0))

    if len(starts) != 4 or len(sizes) != 4:
        raise ValueError(f"Expected 4 SID layers: starts={starts}, sizes={sizes}")
    if beam_size != EXPORT_BEAM_SIZE:
        raise ValueError(
            f"beam size mismatch: config={beam_size}, expected={EXPORT_BEAM_SIZE}"
        )
    return starts, sizes, sid_begin_id, beam_size


def build_prefix_ids(tokenizer, sid_begin_id: int) -> Dict[str, List[int]]:
    result = {}
    for ctype in TARGET_CTYPES:
        text = f"<|ctype_{ctype}|>{SID_BEGIN}"
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 2:
            raise ValueError(f"{text!r} must tokenize to 2 tokens, got {ids}")
        if int(ids[-1]) != sid_begin_id:
            raise ValueError(f"{text!r} does not end with SID_BEGIN id")
        result[ctype] = [int(x) for x in ids]
    return result


# ============================================================================
# residual output -> local SID
# ============================================================================

def decode_pooling_output(output, beam_size: int) -> np.ndarray:
    data = output.outputs.data
    if hasattr(data, "cpu"):
        data = data.cpu().numpy()
    else:
        data = np.asarray(data)

    width = 5  # sid_a,b,c,d + cumulative_score

    if data.ndim == 1:
        if data.size != beam_size * width:
            raise ValueError(
                f"Unexpected pooler output: shape={data.shape}, size={data.size}"
            )
        data = data.reshape(beam_size, width)
    elif data.ndim == 2:
        if data.shape != (beam_size, width):
            raise ValueError(
                f"Unexpected pooler output: shape={data.shape}, "
                f"expected={(beam_size, width)}"
            )
    else:
        raise ValueError(f"Unexpected pooler ndim={data.ndim}, shape={data.shape}")

    # 第 5 列是 cumulative_score。
    # cumulative_score 越大，候选排序越靠前。
    scores = data[:, 4].astype(np.float64, copy=False)

    if not np.all(np.isfinite(scores)):
        raise ValueError(
            "Residual beam scores contain NaN or Inf: "
            f"scores={scores}"
        )

    # 按 cumulative_score 从高到低稳定排序。
    # 分数相同时保持模型原始返回顺序。
    order = np.argsort(
        -scores,
        kind="stable",
    )

    sorted_data = data[order]

    # 排序后只返回前 4 列 global tokenizer token IDs。
    return np.rint(sorted_data[:, :4]).astype(np.int64)


def global_sid_to_local(
    global_sid: Sequence[int],
    starts: Sequence[int],
    sizes: Sequence[int],
) -> Tuple[int, int, int, int]:
    local = []
    for i in range(4):
        token_id = int(global_sid[i])
        code = token_id - int(starts[i])
        if not 0 <= code < int(sizes[i]):
            raise ValueError(
                f"SID token outside layer {i}: token_id={token_id}, "
                f"range=[{starts[i]}, {int(starts[i]) + int(sizes[i])})"
            )
        local.append(code)
    return local[0], local[1], local[2], local[3]


# ============================================================================
# adid2sid reverse map（只由 writer 进程加载一份）
# ============================================================================

def normalize_key(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return ""
        if float(value).is_integer():
            return str(int(value))
    return str(value).strip()


def parse_sid_value(value: Any) -> Tuple[int, int, int, int]:
    if isinstance(value, np.ndarray):
        parts = value.tolist()
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        text = str(value).strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                parts = parsed
            else:
                parts = [x.strip() for x in text.strip("[]").split(",") if x.strip()]
        else:
            parts = [x.strip() for x in text.split(",") if x.strip()]

    if len(parts) != 4:
        raise ValueError(f"Expected 4-layer SID, got {value!r}")
    return tuple(int(x) for x in parts)  # type: ignore[return-value]


def load_sid2adid() -> Dict[Tuple[int, int, int, int], str]:
    print("[Writer] Loading adid2sid reverse map...", flush=True)

    frame = pd.read_parquet(
        ADID2SID_PARQUET,
        columns=["adid", "sid"],
    )

    result: Dict[Tuple[int, int, int, int], str] = {}
    duplicate_same = 0

    for adid, sid in zip(frame["adid"], frame["sid"]):
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
            # 四层 SID 真冲突时无法唯一反查，所以直接报错。
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
    """
    读取：
        /home/jovyan/zhouyuhang-cloud1/sujingsong/ctype_image_size_semid.txt

    默认使用每行前两列：
        原始ctype    映射后的ctype

    例如：
        2    22_180_100

    则最终 output key：
        <mid>_22_180_100
    """
    print(
        f"[Writer] Loading ctype map: {CTYPE_IMAGE_SIZE_SEMID_TXT}",
        flush=True,
    )

    result: Dict[str, str] = {}

    with CTYPE_IMAGE_SIZE_SEMID_TXT.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            cols = line.split()

            if len(cols) < 2:
                raise ValueError(
                    f"Bad ctype map line {line_no}: {line!r}"
                )

            raw_ctype = str(cols[0]).strip()
            mapped_ctype = str(cols[1]).strip()

            old = result.get(raw_ctype)

            if old is not None and old != mapped_ctype:
                raise ValueError(
                    f"ctype={raw_ctype} has multiple mappings: "
                    f"{old!r} vs {mapped_ctype!r}"
                )

            result[raw_ctype] = mapped_ctype

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
    """
    推理过程中先顺序写入一个临时文件。

    全部 GPU worker 完成后，根据最终总行数平均切分成
    NUM_OUTPUT_PARTS 个 part 文件，各 part 行数最多相差 1。
    """

    def __init__(self) -> None:
        self.part_index = 0
        self.current_lines = 0
        self.total_lines = 0
        self.buffer: List[str] = []

        self.temp_path = OUTPUT_DIR / ".all_output.tmp"

        # 防止上一次异常退出留下临时文件。
        if self.temp_path.exists():
            self.temp_path.unlink()

        self.handle = self.temp_path.open(
            "w",
            encoding="utf-8",
            buffering=8 * 1024 * 1024,
        )

        print(
            f"[Writer] Temporary output: {self.temp_path}",
            flush=True,
        )

    def _flush(self) -> None:
        if not self.buffer:
            return

        if self.handle is None:
            raise RuntimeError("Temporary output file is not open")

        self.handle.write("".join(self.buffer))
        self.buffer.clear()

    def append(
        self,
        mid_ctype: str,
        adids: Sequence[str],
    ) -> None:
        """
        一条 (mid, ctype) 只写一行：

            mid_mappedctype<TAB>adid1<TAB>adid2<TAB>...<TAB>adid100

        mid_ctype 和所有 adid 之间全部使用 TAB 分隔。
        """
        if self.handle is None:
            raise RuntimeError("Temporary output file is closed")

        self.buffer.append(
            "\t".join(
                [mid_ctype, *adids]
            )
            + "\n"
        )

        self.current_lines += 1
        self.total_lines += 1

        if len(self.buffer) >= OUTPUT_WRITE_BUFFER_LINES:
            self._flush()

    def close(self) -> None:
        if self.handle is not None:
            self._flush()
            self.handle.close()
            self.handle = None

    def finalize(self) -> None:
        """
        将临时结果平均切分成固定的 NUM_OUTPUT_PARTS 个 part。

        各 part 行数最多相差 1 行。
        当总行数不足 NUM_OUTPUT_PARTS 时，后面的 part 可能为空。
        """
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
            for part_index in range(NUM_OUTPUT_PARTS):
                lines_in_part = (
                    base_lines
                    + (1 if part_index < remainder else 0)
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

            # 正常情况下不应再有剩余数据。
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

def output_writer_worker(output_queue, status_queue, num_gpu_workers: int) -> None:
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
                    # payload:
                    # [(mid, ctype, [(s0,s1,s2,s3), ...]), ...]
                    payload = message[1]

                    for mid, ctype, sid_candidates in payload:
                        prefixes += 1

                        # 保留 residual beam 的原始顺序。
                        # 一个 prefix 的所有 adid 最终写在同一行。
                        adids: List[str] = []

                        for sid_key in sid_candidates:
                            adid = sid2adid.get(tuple(sid_key))

                            if adid is None:
                                sid_miss += 1
                                continue

                            sid_hit += 1
                            adids.append(adid)

                        if not adids:
                            continue

                        mapped_ctype = ctype_output_map.get(ctype)

                        if mapped_ctype is None:
                            raise KeyError(
                                f"ctype={ctype!r} missing in "
                                f"{CTYPE_IMAGE_SIZE_SEMID_TXT}"
                            )

                        mid_ctype = f"{mid}_{mapped_ctype}"

                        writer.append(
                            mid_ctype=mid_ctype,
                            adids=adids,
                        )

                elif kind == "worker_done":
                    done_workers += 1
                    print(
                        f"[Writer] GPU done {done_workers}/{num_gpu_workers}",
                        flush=True,
                    )
                    if done_workers >= num_gpu_workers:
                        break

                elif kind == "abort":
                    raise RuntimeError(f"Abort requested by GPU {message[1]}")

                else:
                    raise ValueError(f"Unknown writer message: {kind}")

        finally:
            writer.close()

        # 只有所有 GPU worker 都正常完成后，
        # 才将临时结果平均切分成 100 个 part。
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
        print(trace, flush=True)
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
    prompt_list: List[Dict[str, Any]],
    prompt_meta: List[Tuple[str, str]],
    beam_size: int,
    starts: Sequence[int],
    sizes: Sequence[int],
    output_queue,
) -> int:
    if len(prompt_list) != len(prompt_meta):
        raise ValueError("prompt/meta length mismatch")

    outputs = llm.encode(prompt_list, use_tqdm=False)

    if len(outputs) != len(prompt_meta):
        raise RuntimeError("vLLM output/meta length mismatch")

    payload = []

    for output, (mid, ctype) in zip(outputs, prompt_meta):
        global_candidates = decode_pooling_output(output, beam_size)

        local_candidates = [
            global_sid_to_local(global_sid, starts, sizes)
            for global_sid in global_candidates
        ]

        payload.append(
            (
                mid,
                ctype,
                local_candidates,
            )
        )

    output_queue.put(("batch", payload))
    return len(outputs)


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
    try:
        # 必须在 import vllm 前设置。
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

        from vllm import LLM

        print(
            f"[Rank {rank}] CUDA_VISIBLE_DEVICES={rank}",
            flush=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        starts, sizes, sid_begin_id, beam_size = load_residual_config(
            model_path
        )

        tokenizer_sid_begin_id = tokenizer.convert_tokens_to_ids(
            SID_BEGIN
        )

        if int(tokenizer_sid_begin_id) != sid_begin_id:
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
            f"starts={starts}, sizes={sizes}",
            flush=True,
        )

        llm = LLM(
            model=model_path,
            task="embed",
            tensor_parallel_size=1,
            dtype=DTYPE,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            trust_remote_code=True,
        )

        parquet_file = pq.ParquetFile(DATA_PATH)

        missing = {"messages", "metadata"} - set(
            parquet_file.schema.names
        )
        if missing:
            raise ValueError(
                f"Input parquet missing columns: {sorted(missing)}"
            )

        # 用 row group 做 8 GPU 分片，避免每个 GPU 全量 pd.read_parquet。
        # 每个 GPU 连续读取一段 row groups。
        # 比 0,8,16... 这种交错读取更接近顺序 I/O，
        # 对超大 Parquet、Ceph/NFS 通常明显更稳定。
        num_row_groups = parquet_file.num_row_groups
        rg_start = (
            num_row_groups * rank
            // world_size
        )
        rg_end = (
            num_row_groups * (rank + 1)
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

        prompt_list: List[Dict[str, Any]] = []
        prompt_meta: List[Tuple[str, str]] = []
        stats: Counter = Counter()

        start_time = time.perf_counter()

        for row_group_index in assigned_row_groups:
            batches = parquet_file.iter_batches(
                batch_size=PARQUET_READ_BATCH_SIZE,
                row_groups=[row_group_index],
                columns=["messages", "metadata"],
                use_threads=PARQUET_USE_THREADS,
            )

            for batch in batches:
                data = batch.to_pydict()

                for messages_value, metadata_value in zip(
                    data["messages"],
                    data["metadata"],
                ):
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

                    # system + user chat template 只做一次。
                    base_ids = tokenizer.apply_chat_template(
                        convert_messages(messages),
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    base_ids = list(base_ids)

                    # 每个符合条件的用户固定构造 3、7、2 三个 prefix prompt。
                    for ctype in TARGET_CTYPES:
                        prompt_token_ids = (
                            base_ids
                            + prefix_ids_map[ctype]
                        )

                        if (
                            not prompt_token_ids
                            or prompt_token_ids[-1] != sid_begin_id
                        ):
                            raise ValueError(
                                "Prompt does not end with SID_BEGIN"
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

                        if len(prompt_list) >= BATCH_SIZE:
                            processed = run_residual_batch(
                                llm=llm,
                                prompt_list=prompt_list,
                                prompt_meta=prompt_meta,
                                beam_size=beam_size,
                                starts=starts,
                                sizes=sizes,
                                output_queue=output_queue,
                            )
                            stats["prefixes_processed"] += processed
                            prompt_list.clear()
                            prompt_meta.clear()

                    if (
                        stats["users_seen"] % PROGRESS_USERS
                        == 0
                    ):
                        elapsed = (
                            time.perf_counter()
                            - start_time
                        )
                        print(
                            f"[Rank {rank}] "
                            f"users={stats['users_seen']:,}; "
                            f"inferred={stats['users_inferred']:,}; "
                            f"skip={stats['users_all_zero_or_no_ctype']:,}; "
                            f"prefixes={stats['prefixes_created']:,}; "
                            f"time={elapsed:.1f}s",
                            flush=True,
                        )

        if prompt_list:
            processed = run_residual_batch(
                llm=llm,
                prompt_list=prompt_list,
                prompt_meta=prompt_meta,
                beam_size=beam_size,
                starts=starts,
                sizes=sizes,
                output_queue=output_queue,
            )
            stats["prefixes_processed"] += processed

        elapsed = (
            time.perf_counter()
            - start_time
        )

        # 同一个 producer 的 batch 会先于 worker_done 进入 Queue。
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
            f"users={stats['users_seen']:,}, "
            f"inferred={stats['users_inferred']:,}, "
            f"prefixes={stats['prefixes_processed']:,}, "
            f"time={elapsed:.2f}s",
            flush=True,
        )

    except Exception as exc:
        trace = traceback.format_exc()
        print(trace, flush=True)

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
            raise FileNotFoundError(path)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def clean_output_parts() -> None:
    # 只清理旧 part-*，不动 OUTPUT_DIR 其他文件。
    for path in OUTPUT_DIR.glob("part-*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def main() -> None:
    check_paths()

    if NUM_GPUS <= 0 or BATCH_SIZE <= 0:
        raise ValueError("NUM_GPUS and BATCH_SIZE must be positive")

    # 只由主进程执行一次 converted -> vLLM085 导出。
    model_path = prepare_vllm_model()

    print("")
    print("=" * 80)
    print("ONLINE RESIDUAL SID INFERENCE")
    print("=" * 80)
    print(f"Converted model : {CONVERTED_MODEL_PATH}")
    print(f"vLLM model      : {model_path}")
    print(f"Input parquet   : {DATA_PATH}")
    print(f"Target CType    : {TARGET_CTYPES}")
    print(f"GPUs            : {NUM_GPUS}")
    print(f"Prefix batch    : {BATCH_SIZE}")
    print(f"Output parts    : {NUM_OUTPUT_PARTS}")
    print(f"Output dir      : {OUTPUT_DIR}")
    print("=" * 80)

    clean_output_parts()

    import multiprocessing as mp

    ctx = mp.get_context("spawn")

    output_queue = ctx.Queue(
        maxsize=OUTPUT_QUEUE_MAXSIZE
    )
    status_queue = ctx.Queue()

    # writer 只加载一份 sid2adid。
    writer_process = ctx.Process(
        target=output_writer_worker,
        args=(
            output_queue,
            status_queue,
            NUM_GPUS,
        ),
        name="result-writer",
    )
    writer_process.start()

    gpu_processes = []

    for rank in range(NUM_GPUS):
        process = ctx.Process(
            target=inference_worker,
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
        gpu_processes.append(process)

    overall_start = time.perf_counter()

    gpu_results = []
    writer_result = None
    failed = None

    try:
        # 正常会收到 NUM_GPUS 个 gpu status + 1 个 writer status。
        while (
            len(gpu_results) < NUM_GPUS
            or writer_result is None
        ):
            status = status_queue.get()

            if not status.get("ok", False):
                failed = status
                break

            if status["kind"] == "gpu":
                gpu_results.append(status)
                print(
                    f"Receive GPU {status['rank']} result",
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
                    f"Unknown status: {status}"
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
                f"{process.name} exitcode={process.exitcode}"
            )

    if writer_process.exitcode != 0:
        raise RuntimeError(
            f"writer exitcode={writer_process.exitcode}"
        )

    if writer_result is None:
        raise RuntimeError("Missing writer result")

    total_stats: Counter = Counter()
    worker_times = []
    beam_size = None

    for result in gpu_results:
        total_stats.update(
            result["stats"]
        )
        worker_times.append(
            float(result["elapsed"])
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
    print(f"Beam size            : {beam_size}")
    print(f"Users seen           : {total_stats['users_seen']:,}")
    print(f"Users inferred       : {total_stats['users_inferred']:,}")
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
    print(f"GPU wall time        : {wall_time:.2f} s")
    print(f"Total elapsed        : {total_elapsed:.2f} s")

    if wall_time > 0:
        print(
            f"User throughput      : "
            f"{total_stats['users_seen'] / wall_time:,.2f} users/s"
        )
        print(
            f"Prefix throughput    : "
            f"{total_stats['prefixes_processed'] / wall_time:,.2f} prefixes/s"
        )

    print(f"Output dir           : {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()