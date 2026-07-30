#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高速版：把 INPUT_DIR 下所有 part-* 合并转换成一个 OpenOneRec-Res
线上推理 Parquet。

输出格式保持不变：
    source
    uuid
    messages
    metadata

主要加速点：
  1. 预先把：
         adidx -> adid -> sid
     合并成：
         adidx -> 已格式化的 SID suffix
     主循环每个历史 item 只做一次 dict lookup。

  2. mid2sid 预先格式化成：
         mid -> <mid_a_*><mid_b_*><mid_c_*>
     主循环不再反复 format。

  3. messages 的 JSON 静态部分只生成一次。
     每行直接拼接 query，不再整棵 json.dumps。

  4. 批量用 Arrow column arrays 写，不再使用
         List[Dict] + pa.Table.from_pylist()
     减少 Python 对象转换开销。

  5. 多个 part-* 使用多进程并行转换为临时 Arrow IPC 文件。
     最后主进程顺序合并成一个大的 Parquet。

  6. 最终 Parquet 只对 source 使用 dictionary encoding。
     uuid/messages/metadata 基本都是高基数字段，不做字典编码。

  7. ZSTD 使用较快的 level=1。

说明：
  - Linux 环境使用 fork，使巨大映射表由 worker 共享 Copy-on-Write 内存，
    避免每个进程重新加载/复制一份。
  - 最终仍然只有一个：
        all_parts_infer.parquet
  - 线上历史不删除最后一个 item。
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import uuid
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq


# ============================================================================
# 固定路径
# ============================================================================

INPUT_DIR = Path(
    "/home/jovyan/zhouyuhang-cloud1/sujingsong/data"
)

PART_GLOB = "part-*"

ADIDX2ADID_TXT = Path(
    "/home/jovyan/ceph-1/zhouyuhang/data/onerec_data/"
    "search_join_dsp_tag_ad_hash_semid.txt.v3"
)

PID2SID_PARQUET = Path(
    "/home/jovyan/ceph-1/sujinsong/online/openonerec-res/"
    "raw_data/onerec_data/adid2sid.parquet"
)

MID2SID_PARQUET = Path(
    "/home/jovyan/ceph-1/sujinsong/online/openonerec-res/"
    "raw_data/onerec_data/mid2sid.parquet"
)

OUTPUT_PARQUET = Path(
    "/home/jovyan/zhouyuhang-cloud1/sujingsong/online_infer/"
    "all_parts_infer.parquet"
)

TEMP_DIR = OUTPUT_PARQUET.parent / ".all_parts_infer_tmp"


# ============================================================================
# 性能配置
# ============================================================================

# 建议先按机器 CPU 核数调整。
# 如果机器内存较紧，可以设成 4；
# CPU/内存充足时可尝试 8 / 12 / 16。
NUM_WORKERS = 16

# 每个 worker 累积多少行后写一个 Arrow RecordBatch。
# 越大通常越快，但单 worker 内存占用也越高。
WRITE_BATCH_SIZE = 100_000

# 最终 Parquet row group 大小。
ROW_GROUP_SIZE = 100_000

# 每个输入 part 每处理多少行打印一次进度。
PROGRESS_INTERVAL = 500_000

# ZSTD 1 明显偏向速度。
PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 1

HIST_MAX_LEN = 512

# 线上场景：最后一条也是历史行为。
DROP_LAST_ITEM_AS_TARGET = False


# ============================================================================
# Token 格式
# ============================================================================

MID_SID_FORMAT = "<mid_a_{t0}><mid_b_{t1}><mid_c_{t2}>"

# ctype 是每条行为动态的，因此这里只预生成 ctype 后面的 SID 部分。
SID_SUFFIX_FORMAT = (
    "<|sid_begin|>"
    "<s_a_{c0}><s_b_{c1}><s_c_{c2}><s_d_{c3}>"
    "<|sid_end|>"
)

SYSTEM_PROMPT = (
    "你是一个智能推荐助手，"
    "能够根据用户的历史行为预测用户可能感兴趣的下一个内容。"
)

USER_PROMPT = (
    "根据以下用户历史行为序列，"
    "请预测用户接下来可能点击的广告：\n{query}"
)

SOURCE = "AdRec_SFT"


# ============================================================================
# 输出 schema
# ============================================================================

OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("source", pa.string()),
        pa.field("uuid", pa.string()),
        pa.field("messages", pa.string()),
        pa.field("metadata", pa.string()),
    ]
)


# ============================================================================
# worker 共享全局映射
#
# 父进程加载完后再 fork。
# Linux fork 下这些大 dict 会通过 Copy-on-Write 共享。
# ============================================================================

_ADIDX2SID_SUFFIX: Dict[str, str] = {}
_MID2TOKEN: Dict[str, str] = {}


# ============================================================================
# 预生成 messages JSON 模板
# ============================================================================

_QUERY_MARKER = "__OPENONEREC_QUERY_MARKER_8F2F40E2__"

_MESSAGE_TEMPLATE = json.dumps(
    [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": USER_PROMPT.format(
                        query=_QUERY_MARKER
                    ),
                }
            ],
        },
    ],
    ensure_ascii=False,
    separators=(",", ":"),
)

if _MESSAGE_TEMPLATE.count(_QUERY_MARKER) != 1:
    raise RuntimeError(
        "Internal messages marker error"
    )

_MESSAGES_PREFIX, _MESSAGES_SUFFIX = (
    _MESSAGE_TEMPLATE.split(
        _QUERY_MARKER,
        1,
    )
)


# ============================================================================
# 基础工具
# ============================================================================

def normalize_key(value: Any) -> str:
    """
    用于加载 Parquet 映射表。

    原始 TSV 本身已经是文本，所以主循环不会频繁调用这个函数。
    """
    if value is None:
        return ""

    if isinstance(value, (np.integer, int)):
        return str(int(value))

    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return ""

        if float(value).is_integer():
            return str(int(value))

    text = str(value).strip()

    if text.endswith(".0"):
        try:
            number = float(text)
        except ValueError:
            return text

        if number.is_integer():
            return str(int(number))

    return text


def parse_sid(
    value: Any,
    expected_layers: int,
) -> Tuple[int, ...]:
    """
    兼容 list / tuple / ndarray / 字符串 sid。
    """
    if isinstance(value, np.ndarray):
        parts = value.tolist()

    elif isinstance(value, (list, tuple)):
        parts = list(value)

    else:
        text = str(value).strip()

        if (
            text.startswith("[")
            and text.endswith("]")
        ):
            try:
                parsed = json.loads(
                    text
                )
            except json.JSONDecodeError:
                parsed = None

            if isinstance(parsed, list):
                parts = parsed
            else:
                parts = [
                    item.strip()
                    for item in text.strip("[]").split(",")
                    if item.strip()
                ]

        else:
            parts = [
                item.strip()
                for item in text.split(",")
                if item.strip()
            ]

    if len(parts) != expected_layers:
        raise ValueError(
            f"SID layer mismatch: "
            f"value={value!r}, "
            f"expected={expected_layers}, "
            f"actual={len(parts)}"
        )

    return tuple(
        int(item)
        for item in parts
    )


def build_messages_fast(
    query: str,
) -> str:
    """
    query 只由模型特殊 token 组成：
      <mid_*>
      <|ctype_*|>
      <|sid_begin|>
      <s_*>
      <|sid_end|>

    不含双引号、反斜杠或真实换行，
    因此可以安全插入已经 json.dumps 好的模板。
    """
    return (
        _MESSAGES_PREFIX
        + query
        + _MESSAGES_SUFFIX
    )


def build_metadata_fast(
    mid: str,
) -> str:
    """
    mid 正常情况下是数字字符串。

    仍使用 json.dumps(mid) 保证即使出现特殊字符，
    metadata 也始终是合法 JSON。
    """
    return (
        '{"mid":'
        + json.dumps(
            mid,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "}"
    )


# ============================================================================
# 映射加载与预合并
# ============================================================================

def load_adid2sid_suffix() -> Dict[str, str]:
    """
    adid -> 已经格式化的 SID suffix

    例如：
      12345 ->
      <|sid_begin|><s_a_1><s_b_2><s_c_3><s_d_4><|sid_end|>
    """
    frame = pd.read_parquet(
        PID2SID_PARQUET,
        columns=[
            "adid",
            "sid",
        ],
    )

    result: Dict[str, str] = {}

    for adid, sid in zip(
        frame["adid"],
        frame["sid"],
    ):
        key = normalize_key(
            adid
        )

        if not key:
            continue

        code = parse_sid(
            sid,
            expected_layers=4,
        )

        result[key] = (
            SID_SUFFIX_FORMAT.format(
                c0=code[0],
                c1=code[1],
                c2=code[2],
                c3=code[3],
            )
        )

    return result


def load_adidx2sid_suffix(
    adid2sid_suffix: Dict[str, str],
) -> Tuple[Dict[str, str], Counter]:
    """
    一次性把两层映射：
        adidx -> adid
        adid  -> sid

    合并成运行时只需要的一层：
        adidx -> formatted SID suffix
    """
    result: Dict[str, str] = {}
    stats: Counter = Counter()

    with ADIDX2ADID_TXT.open(
        "r",
        encoding="utf-8",
        errors="ignore",
        buffering=1024 * 1024,
    ) as stream:
        for line in stream:
            cols = line.rstrip(
                "\r\n"
            ).split(
                "\t",
                2,
            )

            if len(cols) < 2:
                stats["malformed_mapping_rows"] += 1
                continue

            adidx = cols[0].strip()
            adid = cols[1].strip()

            if not adidx or not adid:
                stats["malformed_mapping_rows"] += 1
                continue

            stats["adidx2adid_rows"] += 1

            sid_suffix = (
                adid2sid_suffix.get(
                    adid
                )
            )

            if sid_suffix is None:
                stats["adid_without_sid"] += 1
                continue

            result[adidx] = (
                sid_suffix
            )

            stats["adidx_with_sid"] += 1

    return (
        result,
        stats,
    )


def load_mid2token() -> Dict[str, str]:
    """
    mid -> 已经格式化的 MID token。
    """
    frame = pd.read_parquet(
        MID2SID_PARQUET,
        columns=[
            "mid",
            "sid",
        ],
    )

    result: Dict[str, str] = {}

    for mid, sid in zip(
        frame["mid"],
        frame["sid"],
    ):
        key = normalize_key(
            mid
        )

        if not key:
            continue

        code = parse_sid(
            sid,
            expected_layers=3,
        )

        result[key] = (
            MID_SID_FORMAT.format(
                t0=code[0],
                t1=code[1],
                t2=code[2],
            )
        )

    return result


# ============================================================================
# 输入发现
# ============================================================================

def discover_input_parts() -> List[Path]:
    parts = sorted(
        path
        for path in INPUT_DIR.glob(
            PART_GLOB
        )
        if path.is_file()
    )

    if not parts:
        raise FileNotFoundError(
            f"No files matching "
            f"{PART_GLOB!r} "
            f"under {INPUT_DIR}"
        )

    return parts


def check_paths() -> None:
    if not INPUT_DIR.is_dir():
        raise NotADirectoryError(
            f"INPUT_DIR does not exist "
            f"or is not a directory: "
            f"{INPUT_DIR}"
        )

    for path in [
        ADIDX2ADID_TXT,
        PID2SID_PARQUET,
        MID2SID_PARQUET,
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required file does not exist: "
                f"{path}"
            )

    OUTPUT_PARQUET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================================
# 热路径：单行处理
# ============================================================================

def process_line_fast(
    line: str,
    stats: Counter,
    ctype_prefix_cache: Dict[str, str],
) -> Tuple[str, str, str, str] | None:
    """
    热路径尽量减少函数调用和中间 Python 对象。

    返回：
        source, uuid, messages, metadata
    """
    # 最多只关心第 6 列。
    # maxsplit=6 可以避免不必要地切后面的字段。
    cols = line.rstrip(
        "\r\n"
    ).split(
        "\t",
        6,
    )

    if len(cols) < 6:
        stats["bad_column_rows"] += 1
        return None

    mid = cols[0].strip()

    if not mid:
        stats["empty_mid_rows"] += 1
        return None

    adidx_text = cols[2].strip()
    ctype_text = cols[5].strip()

    if not adidx_text:
        stats["empty_ad_sequence_rows"] += 1
        return None

    if not ctype_text:
        stats["empty_ctype_sequence_rows"] += 1
        return None

    adidxs = adidx_text.split(
        ","
    )

    ctypeids = ctype_text.split(
        ","
    )

    pair_count = min(
        len(adidxs),
        len(ctypeids),
    )

    if (
        len(adidxs)
        != len(ctypeids)
    ):
        stats[
            "sequence_length_mismatch_rows"
        ] += 1

    if pair_count <= 0:
        stats["empty_pair_rows"] += 1
        return None

    start_index = max(
        0,
        pair_count - HIST_MAX_LEN,
    )

    if DROP_LAST_ITEM_AS_TARGET:
        pair_count -= 1

        if pair_count <= 0:
            stats["empty_pair_rows"] += 1
            return None

        start_index = max(
            0,
            pair_count - HIST_MAX_LEN,
        )

    pieces: List[str] = []

    mid_token = _MID2TOKEN.get(
        mid
    )

    if mid_token is not None:
        pieces.append(
            mid_token
        )
        stats["mid_sid_hit"] += 1
    else:
        stats["mid_sid_miss"] += 1

    item_hit = 0

    # 局部变量绑定，减少循环里的全局查找。
    adidx_map_get = (
        _ADIDX2SID_SUFFIX.get
    )

    prefix_cache_get = (
        ctype_prefix_cache.get
    )

    prefix_cache_set = (
        ctype_prefix_cache.__setitem__
    )

    for i in range(
        start_index,
        pair_count,
    ):
        adidx = adidxs[i].strip()

        if not adidx:
            stats["empty_adidx_items"] += 1
            continue

        sid_suffix = (
            adidx_map_get(
                adidx
            )
        )

        if sid_suffix is None:
            stats["item_sid_miss"] += 1
            continue

        ctype = ctypeids[i].strip()

        if not ctype:
            stats["empty_ctype_items"] += 1
            continue

        ctype_prefix = (
            prefix_cache_get(
                ctype
            )
        )

        if ctype_prefix is None:
            ctype_prefix = (
                "<|ctype_"
                + ctype
                + "|>"
            )

            prefix_cache_set(
                ctype,
                ctype_prefix,
            )

        pieces.append(
            ctype_prefix
            + sid_suffix
        )

        item_hit += 1
        stats["item_sid_hit"] += 1

    # 保持原逻辑：
    # 只要 MID 命中，即使 item 全 miss，query 仍然非空。
    if not pieces:
        stats["empty_query_rows"] += 1
        return None

    query = "".join(
        pieces
    )

    stats["output_rows"] += 1

    return (
        SOURCE,
        str(uuid.uuid4()),
        build_messages_fast(
            query
        ),
        build_metadata_fast(
            mid
        ),
    )


# ============================================================================
# Arrow batch
# ============================================================================

def write_ipc_batch(
    writer: ipc.RecordBatchFileWriter,
    source_values: List[str],
    uuid_values: List[str],
    messages_values: List[str],
    metadata_values: List[str],
) -> None:
    if not source_values:
        return

    batch = pa.record_batch(
        [
            pa.array(
                source_values,
                type=pa.string(),
            ),
            pa.array(
                uuid_values,
                type=pa.string(),
            ),
            pa.array(
                messages_values,
                type=pa.string(),
            ),
            pa.array(
                metadata_values,
                type=pa.string(),
            ),
        ],
        schema=OUTPUT_SCHEMA,
    )

    writer.write_batch(
        batch
    )

    source_values.clear()
    uuid_values.clear()
    messages_values.clear()
    metadata_values.clear()


# ============================================================================
# Worker：每个 part -> 临时 Arrow IPC
# ============================================================================

def process_part_worker(
    input_part_str: str,
    temp_file_str: str,
) -> Dict[str, Any]:
    input_part = Path(
        input_part_str
    )

    temp_file = Path(
        temp_file_str
    )

    stats: Counter = Counter()

    source_values: List[str] = []
    uuid_values: List[str] = []
    messages_values: List[str] = []
    metadata_values: List[str] = []

    ctype_prefix_cache: Dict[
        str,
        str
    ] = {}

    input_rows = 0
    skipped_rows = 0

    start_time = (
        time.perf_counter()
    )

    with pa.OSFile(
        str(temp_file),
        "wb",
    ) as sink:
        with ipc.new_file(
            sink,
            OUTPUT_SCHEMA,
        ) as writer:
            with input_part.open(
                "r",
                encoding="utf-8",
                errors="ignore",
                buffering=4 * 1024 * 1024,
            ) as stream:
                for line in stream:
                    input_rows += 1

                    result = process_line_fast(
                        line=line,
                        stats=stats,
                        ctype_prefix_cache=(
                            ctype_prefix_cache
                        ),
                    )

                    if result is None:
                        skipped_rows += 1
                        continue

                    (
                        source,
                        row_uuid,
                        messages,
                        metadata,
                    ) = result

                    source_values.append(
                        source
                    )
                    uuid_values.append(
                        row_uuid
                    )
                    messages_values.append(
                        messages
                    )
                    metadata_values.append(
                        metadata
                    )

                    if (
                        len(source_values)
                        >= WRITE_BATCH_SIZE
                    ):
                        write_ipc_batch(
                            writer=writer,
                            source_values=source_values,
                            uuid_values=uuid_values,
                            messages_values=messages_values,
                            metadata_values=metadata_values,
                        )

                    if (
                        input_rows
                        % PROGRESS_INTERVAL
                        == 0
                    ):
                        print(
                            f"[pid={os.getpid()}] "
                            f"{input_part.name}: "
                            f"processed={input_rows:,}, "
                            f"output={stats['output_rows']:,}, "
                            f"skipped={skipped_rows:,}",
                            flush=True,
                        )

            write_ipc_batch(
                writer=writer,
                source_values=source_values,
                uuid_values=uuid_values,
                messages_values=messages_values,
                metadata_values=metadata_values,
            )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    return {
        "part_name": input_part.name,
        "temp_file": str(
            temp_file
        ),
        "input_rows": input_rows,
        "output_rows": int(
            stats["output_rows"]
        ),
        "skipped_rows": skipped_rows,
        "elapsed": elapsed,
        "stats": dict(stats),
    }


# ============================================================================
# 合并临时 Arrow -> 一个大 Parquet
# ============================================================================

def merge_temp_arrow_files(
    input_parts: Sequence[Path],
    temp_paths: Dict[str, Path],
) -> None:
    if OUTPUT_PARQUET.exists():
        OUTPUT_PARQUET.unlink()

    writer = pq.ParquetWriter(
        OUTPUT_PARQUET,
        schema=OUTPUT_SCHEMA,
        compression=(
            PARQUET_COMPRESSION
        ),
        compression_level=(
            PARQUET_COMPRESSION_LEVEL
        ),
        # source 只有一个固定值；
        # 其余三列高基数，不适合 dictionary。
        use_dictionary=[
            "source",
        ],
        write_statistics=True,
    )

    try:
        for file_index, input_part in enumerate(
            input_parts,
            start=1,
        ):
            temp_path = temp_paths[
                input_part.name
            ]

            print(
                f"[merge "
                f"{file_index}/{len(input_parts)}] "
                f"{input_part.name}",
                flush=True,
            )

            with pa.memory_map(
                str(temp_path),
                "r",
            ) as source:
                reader = (
                    ipc.open_file(
                        source
                    )
                )

                for batch_index in range(
                    reader.num_record_batches
                ):
                    batch = (
                        reader.get_batch(
                            batch_index
                        )
                    )

                    writer.write_table(
                        pa.Table.from_batches(
                            [batch],
                            schema=OUTPUT_SCHEMA,
                        ),
                        row_group_size=(
                            ROW_GROUP_SIZE
                        ),
                    )

    finally:
        writer.close()


# ============================================================================
# Preview
# ============================================================================

def print_mapping_preview(
    input_part: Path,
) -> None:
    print("")
    print(
        f"Mapping preview from "
        f"{input_part.name}:"
    )

    printed = 0

    with input_part.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as stream:
        for line in stream:
            cols = line.rstrip(
                "\r\n"
            ).split(
                "\t",
                6,
            )

            if len(cols) < 6:
                continue

            adidxs = (
                cols[2]
                .strip()
                .split(",")
            )

            for adidx in adidxs:
                adidx = adidx.strip()

                if not adidx:
                    continue

                sid_hit = (
                    adidx
                    in _ADIDX2SID_SUFFIX
                )

                print(
                    f"  adidx={adidx} "
                    f"-> sid_hit={sid_hit}"
                )

                printed += 1

                if (
                    printed
                    >= 10
                ):
                    return


def print_first_output_row() -> None:
    parquet_file = (
        pq.ParquetFile(
            OUTPUT_PARQUET
        )
    )

    if (
        parquet_file.metadata.num_rows
        <= 0
    ):
        print(
            "Output parquet is empty."
        )
        return

    table = (
        parquet_file
        .read_row_group(
            0,
            columns=[
                "source",
                "uuid",
                "messages",
                "metadata",
            ],
        )
        .slice(
            0,
            1,
        )
    )

    row = (
        table
        .to_pylist()[0]
    )

    print("")
    print("First row:")
    print(
        f"source   : "
        f"{row['source']}"
    )
    print(
        f"uuid     : "
        f"{row['uuid']}"
    )
    print(
        f"messages : "
        f"{row['messages']}"
    )
    print(
        f"metadata : "
        f"{row['metadata']}"
    )


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    global _ADIDX2SID_SUFFIX
    global _MID2TOKEN

    check_paths()

    input_parts = (
        discover_input_parts()
    )

    print(
        f"Input dir      : "
        f"{INPUT_DIR}"
    )
    print(
        f"Found parts    : "
        f"{len(input_parts):,}"
    )
    print(
        f"Output         : "
        f"{OUTPUT_PARQUET}"
    )
    print(
        f"Workers        : "
        f"{NUM_WORKERS}"
    )
    print(
        f"Batch size     : "
        f"{WRITE_BATCH_SIZE:,}"
    )

    overall_start = (
        time.perf_counter()
    )

    # ------------------------------------------------------------------------
    # 1. 加载并合并 item 映射
    # ------------------------------------------------------------------------

    print("")
    print(
        "Loading adid -> SID suffix..."
    )

    adid2sid_suffix = (
        load_adid2sid_suffix()
    )

    print(
        f"Loaded adid SID: "
        f"{len(adid2sid_suffix):,}"
    )

    print(
        "Building adidx -> SID suffix..."
    )

    (
        _ADIDX2SID_SUFFIX,
        mapping_stats,
    ) = load_adidx2sid_suffix(
        adid2sid_suffix
    )

    # 合并完立即释放中间大 dict。
    del adid2sid_suffix

    print(
        f"Built adidx SID: "
        f"{len(_ADIDX2SID_SUFFIX):,}"
    )
    print(
        f"adidx rows      : "
        f"{mapping_stats['adidx2adid_rows']:,}"
    )
    print(
        f"adid no SID     : "
        f"{mapping_stats['adid_without_sid']:,}"
    )

    # ------------------------------------------------------------------------
    # 2. 加载并预格式化 MID
    # ------------------------------------------------------------------------

    print(
        "Loading mid -> token..."
    )

    _MID2TOKEN = (
        load_mid2token()
    )

    print(
        f"Loaded MID token: "
        f"{len(_MID2TOKEN):,}"
    )

    print_mapping_preview(
        input_parts[0]
    )

    # ------------------------------------------------------------------------
    # 3. 清理临时目录
    # ------------------------------------------------------------------------

    if TEMP_DIR.exists():
        shutil.rmtree(
            TEMP_DIR
        )

    TEMP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_paths: Dict[
        str,
        Path
    ] = {}

    for input_part in input_parts:
        temp_paths[
            input_part.name
        ] = (
            TEMP_DIR
            / (
                input_part.name
                + ".arrow"
            )
        )

    # ------------------------------------------------------------------------
    # 4. 多进程并行转换
    # ------------------------------------------------------------------------

    worker_count = min(
        NUM_WORKERS,
        len(input_parts),
    )

    if worker_count <= 0:
        raise ValueError(
            "worker_count must be positive"
        )

    print("")
    print(
        f"Parallel converting with "
        f"{worker_count} workers..."
    )

    # 当前服务器路径显然是 Linux。
    # fork 的目的：共享父进程中已经加载好的大映射 dict。
    ctx = mp.get_context(
        "fork"
    )

    results: Dict[
        str,
        Dict[str, Any]
    ] = {}

    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=ctx,
        ) as executor:
            future_to_part = {}

            for input_part in input_parts:
                future = executor.submit(
                    process_part_worker,
                    str(input_part),
                    str(
                        temp_paths[
                            input_part.name
                        ]
                    ),
                )

                future_to_part[
                    future
                ] = input_part

            completed = 0

            for future in as_completed(
                future_to_part
            ):
                input_part = (
                    future_to_part[
                        future
                    ]
                )

                result = (
                    future.result()
                )

                results[
                    input_part.name
                ] = result

                completed += 1

                print(
                    f"[done "
                    f"{completed}/{len(input_parts)}] "
                    f"{input_part.name}: "
                    f"input="
                    f"{result['input_rows']:,}, "
                    f"output="
                    f"{result['output_rows']:,}, "
                    f"skipped="
                    f"{result['skipped_rows']:,}, "
                    f"time="
                    f"{result['elapsed']:.2f}s",
                    flush=True,
                )

        # --------------------------------------------------------------------
        # 5. 合并为一个 Parquet
        # --------------------------------------------------------------------

        print("")
        print(
            "Merging temporary Arrow files "
            "into one Parquet..."
        )

        merge_start = (
            time.perf_counter()
        )

        merge_temp_arrow_files(
            input_parts=input_parts,
            temp_paths=temp_paths,
        )

        merge_elapsed = (
            time.perf_counter()
            - merge_start
        )

        # --------------------------------------------------------------------
        # 6. 汇总统计
        # --------------------------------------------------------------------

        total_input = sum(
            int(
                result["input_rows"]
            )
            for result in results.values()
        )

        total_output = sum(
            int(
                result["output_rows"]
            )
            for result in results.values()
        )

        total_skipped = sum(
            int(
                result["skipped_rows"]
            )
            for result in results.values()
        )

        total_stats: Counter = Counter()

        for result in results.values():
            total_stats.update(
                result["stats"]
            )

        overall_elapsed = (
            time.perf_counter()
            - overall_start
        )

        throughput = (
            total_input
            / overall_elapsed
            if overall_elapsed > 0
            else 0.0
        )

        output_size = (
            OUTPUT_PARQUET.stat().st_size
            if OUTPUT_PARQUET.exists()
            else 0
        )

        print("")
        print("=" * 72)
        print("ALL DONE")
        print("=" * 72)
        print(
            f"Files processed      : "
            f"{len(input_parts):,}"
        )
        print(
            f"Input rows           : "
            f"{total_input:,}"
        )
        print(
            f"Output rows          : "
            f"{total_output:,}"
        )
        print(
            f"Skipped rows         : "
            f"{total_skipped:,}"
        )
        print(
            f"Item SID hit         : "
            f"{total_stats['item_sid_hit']:,}"
        )
        print(
            f"Item SID miss        : "
            f"{total_stats['item_sid_miss']:,}"
        )
        print(
            f"MID hit              : "
            f"{total_stats['mid_sid_hit']:,}"
        )
        print(
            f"MID miss             : "
            f"{total_stats['mid_sid_miss']:,}"
        )
        print(
            f"Length mismatch rows : "
            f"{total_stats['sequence_length_mismatch_rows']:,}"
        )
        print(
            f"Merge time           : "
            f"{merge_elapsed:.2f}s"
        )
        print(
            f"Total time           : "
            f"{overall_elapsed:.2f}s"
        )
        print(
            f"Input throughput     : "
            f"{throughput:,.2f} rows/s"
        )
        print(
            f"Output size          : "
            f"{output_size / 1024 / 1024 / 1024:.2f} GiB"
        )
        print(
            f"Saved                : "
            f"{OUTPUT_PARQUET}"
        )
        print("=" * 72)

        print_first_output_row()

    finally:
        # 成功或失败都尽量清理临时 Arrow。
        if TEMP_DIR.exists():
            shutil.rmtree(
                TEMP_DIR,
                ignore_errors=True,
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        raise
