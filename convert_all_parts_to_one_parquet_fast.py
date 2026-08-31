#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenOneRec-Res online raw part-* -> one inference parquet (add_feature + time).

Online query protocol:
    MID
    + (CType + SID + TIME) * history

TIME semantics:
1) DROP_LAST_ITEM_AS_TARGET=True:
       anchor = the dropped last item's timestamp
   This is the same as training:
       time = first_target_time - hist_time

2) DROP_LAST_ITEM_AS_TARGET=False (current online setting):
       no future target timestamp exists at serving time, so
       anchor = the latest observed item's timestamp
       time = anchor - hist_time

Bucket:
    floor(delta_seconds / 3600), clipped to [0, 336]

Raw input columns follow the current online/training raw schema:
    col[0] mid
    col[2] adidx sequence
    col[4] timestamp sequence
    col[5] ctype sequence
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
# Fixed paths: kept aligned with the current online script.
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
# Performance configuration: preserved from the current online version.
# ============================================================================

NUM_WORKERS = 16
WRITE_BATCH_SIZE = 100_000
ROW_GROUP_SIZE = 100_000
PROGRESS_INTERVAL = 500_000

PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 1

HIST_MAX_LEN = 512

# Current online semantics: the last observed item is still history.
DROP_LAST_ITEM_AS_TARGET = False


# ============================================================================
# Token formats
# ============================================================================

MID_SID_FORMAT = "<mid_a_{t0}><mid_b_{t1}><mid_c_{t2}>"

SID_SUFFIX_FORMAT = (
    "<|sid_begin|>"
    "<s_a_{c0}><s_b_{c1}><s_c_{c2}><s_d_{c3}>"
    "<|sid_end|>"
)

TIME_FORMAT = "<|time_{t0}|>"
MAX_TIME_BUCKET = 336

# Precreate all legal TIME strings.
_TIME_TOKENS: Tuple[str, ...] = tuple(
    TIME_FORMAT.format(t0=i)
    for i in range(MAX_TIME_BUCKET + 1)
)

SYSTEM_PROMPT = (
    "你是一个智能推荐助手，"
    "能够根据用户的历史行为预测用户可能感兴趣的下一个内容。"
)

USER_PROMPT = (
    "根据以下用户历史行为序列，"
    "请预测用户接下来可能点击的广告：\n{query}"
)

# Keep the old source name to avoid changing downstream source filters.
SOURCE = "AdRec_SFT"


# ============================================================================
# Output schema
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
# Worker-shared maps
# ============================================================================

_ADIDX2SID_SUFFIX: Dict[str, str] = {}
_MID2TOKEN: Dict[str, str] = {}


# ============================================================================
# Fast messages template
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
                    "text": USER_PROMPT.format(query=_QUERY_MARKER),
                }
            ],
        },
    ],
    ensure_ascii=False,
    separators=(",", ":"),
)

if _MESSAGE_TEMPLATE.count(_QUERY_MARKER) != 1:
    raise RuntimeError("Internal messages marker error")

_MESSAGES_PREFIX, _MESSAGES_SUFFIX = _MESSAGE_TEMPLATE.split(
    _QUERY_MARKER,
    1,
)


# ============================================================================
# Basic helpers
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


def parse_sid(
    value: Any,
    expected_layers: int,
) -> Tuple[int, ...]:
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
            "SID layer mismatch: "
            f"value={value!r}, expected={expected_layers}, actual={len(parts)}"
        )

    return tuple(int(item) for item in parts)


def parse_timestamp_seconds(
    value: str,
) -> int:
    """
    Raw training code uses // 3600, therefore online timestamps must use
    the same seconds unit.  float -> int is accepted for strings such as
    '1720000000.0'.
    """
    text = value.strip()
    if not text:
        raise ValueError("empty timestamp")
    number = float(text)
    if not np.isfinite(number):
        raise ValueError(f"non-finite timestamp: {value!r}")
    return int(number)


def time_bucket(
    hist_time: int,
    anchor_time: int,
    stats: Counter,
) -> int:
    delta = anchor_time - hist_time

    if delta < 0:
        # Serving data may occasionally be slightly unordered/dirty.
        # The tokenizer only contains time_0 ... time_336, so never emit
        # a negative token.
        stats["negative_time_delta_items"] += 1

    bucket = delta // 3600

    if bucket < 0:
        bucket = 0
    elif bucket > MAX_TIME_BUCKET:
        bucket = MAX_TIME_BUCKET
        stats["time_bucket_clamped_items"] += 1

    return int(bucket)


def build_messages_fast(
    query: str,
) -> str:
    # query only contains model special-token strings.
    return _MESSAGES_PREFIX + query + _MESSAGES_SUFFIX


def build_metadata_fast(
    mid: str,
) -> str:
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
# Mapping loading
# ============================================================================

def load_adid2sid_suffix() -> Dict[str, str]:
    frame = pd.read_parquet(
        PID2SID_PARQUET,
        columns=["adid", "sid"],
    )

    result: Dict[str, str] = {}

    for adid, sid in zip(
        frame["adid"],
        frame["sid"],
    ):
        key = normalize_key(adid)
        if not key:
            continue

        code = parse_sid(
            sid,
            expected_layers=4,
        )

        result[key] = SID_SUFFIX_FORMAT.format(
            c0=code[0],
            c1=code[1],
            c2=code[2],
            c3=code[3],
        )

    return result


def load_adidx2sid_suffix(
    adid2sid_suffix: Dict[str, str],
) -> Tuple[Dict[str, str], Counter]:
    result: Dict[str, str] = {}
    stats: Counter = Counter()

    with ADIDX2ADID_TXT.open(
        "r",
        encoding="utf-8",
        errors="ignore",
        buffering=1024 * 1024,
    ) as stream:
        for line in stream:
            cols = line.rstrip("\r\n").split("\t", 2)

            if len(cols) < 2:
                stats["malformed_mapping_rows"] += 1
                continue

            adidx = cols[0].strip()
            adid = cols[1].strip()

            if not adidx or not adid:
                stats["malformed_mapping_rows"] += 1
                continue

            stats["adidx2adid_rows"] += 1

            sid_suffix = adid2sid_suffix.get(adid)

            if sid_suffix is None:
                stats["adid_without_sid"] += 1
                continue

            result[adidx] = sid_suffix
            stats["adidx_with_sid"] += 1

    return result, stats


def load_mid2token() -> Dict[str, str]:
    frame = pd.read_parquet(
        MID2SID_PARQUET,
        columns=["mid", "sid"],
    )

    result: Dict[str, str] = {}

    for mid, sid in zip(
        frame["mid"],
        frame["sid"],
    ):
        key = normalize_key(mid)
        if not key:
            continue

        code = parse_sid(
            sid,
            expected_layers=3,
        )

        result[key] = MID_SID_FORMAT.format(
            t0=code[0],
            t1=code[1],
            t2=code[2],
        )

    return result


# ============================================================================
# Input discovery / validation
# ============================================================================

def discover_input_parts() -> List[Path]:
    parts = sorted(
        path
        for path in INPUT_DIR.glob(PART_GLOB)
        if path.is_file()
    )

    if not parts:
        raise FileNotFoundError(
            f"No files matching {PART_GLOB!r} under {INPUT_DIR}"
        )

    return parts


def check_paths() -> None:
    if not INPUT_DIR.is_dir():
        raise NotADirectoryError(
            f"INPUT_DIR does not exist or is not a directory: {INPUT_DIR}"
        )

    for path in (
        ADIDX2ADID_TXT,
        PID2SID_PARQUET,
        MID2SID_PARQUET,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required file does not exist: {path}"
            )

    OUTPUT_PARQUET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================================
# Hot path: one raw row
# ============================================================================

def process_line_fast(
    line: str,
    stats: Counter,
    ctype_prefix_cache: Dict[str, str],
) -> Tuple[str, str, str, str] | None:
    """
    Return:
        source, uuid, messages, metadata

    Raw columns used:
        0: mid
        2: adidx sequence
        4: timestamp sequence
        5: ctype sequence
    """
    cols = line.rstrip("\r\n").split("\t", 6)

    if len(cols) < 6:
        stats["bad_column_rows"] += 1
        return None

    mid = cols[0].strip()
    if not mid:
        stats["empty_mid_rows"] += 1
        return None

    adidx_text = cols[2].strip()
    time_text = cols[4].strip()
    ctype_text = cols[5].strip()

    if not adidx_text:
        stats["empty_ad_sequence_rows"] += 1
        return None

    if not time_text:
        stats["empty_time_sequence_rows"] += 1
        return None

    if not ctype_text:
        stats["empty_ctype_sequence_rows"] += 1
        return None

    adidxs = adidx_text.split(",")
    timeids = time_text.split(",")
    ctypeids = ctype_text.split(",")

    item_count = min(
        len(adidxs),
        len(timeids),
        len(ctypeids),
    )

    if not (
        len(adidxs)
        == len(timeids)
        == len(ctypeids)
    ):
        stats["sequence_length_mismatch_rows"] += 1

    if item_count <= 0:
        stats["empty_pair_rows"] += 1
        return None

    # ------------------------------------------------------------
    # Prediction anchor:
    #
    # DROP_LAST_ITEM_AS_TARGET=True:
    #   [history ...] [target]
    #   anchor = target timestamp, history excludes last.
    #
    # DROP_LAST_ITEM_AS_TARGET=False:
    #   everything is observed history.
    #   anchor = latest observed timestamp.
    # ------------------------------------------------------------
    anchor_index = item_count - 1

    try:
        anchor_time = parse_timestamp_seconds(
            timeids[anchor_index]
        )
    except Exception:
        stats["bad_anchor_time_rows"] += 1
        return None

    history_end = (
        item_count - 1
        if DROP_LAST_ITEM_AS_TARGET
        else item_count
    )

    if history_end <= 0:
        stats["empty_pair_rows"] += 1
        return None

    start_index = max(
        0,
        history_end - HIST_MAX_LEN,
    )

    pieces: List[str] = []

    mid_token = _MID2TOKEN.get(mid)

    if mid_token is not None:
        pieces.append(mid_token)
        stats["mid_sid_hit"] += 1
    else:
        stats["mid_sid_miss"] += 1

    adidx_map_get = _ADIDX2SID_SUFFIX.get
    prefix_cache_get = ctype_prefix_cache.get
    prefix_cache_set = ctype_prefix_cache.__setitem__

    for i in range(
        start_index,
        history_end,
    ):
        adidx = adidxs[i].strip()

        if not adidx:
            stats["empty_adidx_items"] += 1
            continue

        sid_suffix = adidx_map_get(adidx)

        if sid_suffix is None:
            stats["item_sid_miss"] += 1
            continue

        ctype = ctypeids[i].strip()

        if not ctype:
            stats["empty_ctype_items"] += 1
            continue

        try:
            hist_time = parse_timestamp_seconds(
                timeids[i]
            )
        except Exception:
            stats["bad_time_items"] += 1
            continue

        bucket = time_bucket(
            hist_time=hist_time,
            anchor_time=anchor_time,
            stats=stats,
        )

        ctype_prefix = prefix_cache_get(ctype)

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
            + _TIME_TOKENS[bucket]
        )

        stats["item_sid_hit"] += 1
        stats["time_token_items"] += 1

    # Preserve the old behavior: a MID-only row can still exist.
    # The inference script will simply skip it because it has no CType.
    if not pieces:
        stats["empty_query_rows"] += 1
        return None

    query = "".join(pieces)

    stats["output_rows"] += 1

    return (
        SOURCE,
        str(uuid.uuid4()),
        build_messages_fast(query),
        build_metadata_fast(mid),
    )


# ============================================================================
# Arrow IPC batching
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
            pa.array(source_values, type=pa.string()),
            pa.array(uuid_values, type=pa.string()),
            pa.array(messages_values, type=pa.string()),
            pa.array(metadata_values, type=pa.string()),
        ],
        schema=OUTPUT_SCHEMA,
    )

    writer.write_batch(batch)

    source_values.clear()
    uuid_values.clear()
    messages_values.clear()
    metadata_values.clear()


# ============================================================================
# Worker: one part -> temporary Arrow IPC
# ============================================================================

def process_part_worker(
    input_part_str: str,
    temp_file_str: str,
) -> Dict[str, Any]:
    input_part = Path(input_part_str)
    temp_file = Path(temp_file_str)

    stats: Counter = Counter()

    source_values: List[str] = []
    uuid_values: List[str] = []
    messages_values: List[str] = []
    metadata_values: List[str] = []

    ctype_prefix_cache: Dict[str, str] = {}

    input_rows = 0
    skipped_rows = 0

    start_time = time.perf_counter()

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
                        ctype_prefix_cache=ctype_prefix_cache,
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

                    source_values.append(source)
                    uuid_values.append(row_uuid)
                    messages_values.append(messages)
                    metadata_values.append(metadata)

                    if len(source_values) >= WRITE_BATCH_SIZE:
                        write_ipc_batch(
                            writer=writer,
                            source_values=source_values,
                            uuid_values=uuid_values,
                            messages_values=messages_values,
                            metadata_values=metadata_values,
                        )

                    if (
                        input_rows % PROGRESS_INTERVAL
                        == 0
                    ):
                        print(
                            f"[pid={os.getpid()}] "
                            f"{input_part.name}: "
                            f"processed={input_rows:,}, "
                            f"output={stats['output_rows']:,}, "
                            f"skipped={skipped_rows:,}, "
                            f"time_tokens={stats['time_token_items']:,}",
                            flush=True,
                        )

            write_ipc_batch(
                writer=writer,
                source_values=source_values,
                uuid_values=uuid_values,
                messages_values=messages_values,
                metadata_values=metadata_values,
            )

    elapsed = time.perf_counter() - start_time

    return {
        "part_name": input_part.name,
        "temp_file": str(temp_file),
        "input_rows": input_rows,
        "output_rows": int(stats["output_rows"]),
        "skipped_rows": skipped_rows,
        "elapsed": elapsed,
        "stats": dict(stats),
    }


# ============================================================================
# Merge temporary Arrow -> one parquet
# ============================================================================

def merge_temp_arrow_files(
    input_parts: Sequence[Path],
    temp_paths: Dict[str, Path],
) -> None:
    if OUTPUT_PARQUET.exists():
        OUTPUT_PARQUET.unlink()

    writer = pq.ParquetWriter(
        OUTPUT_PARQUET,
        OUTPUT_SCHEMA,
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
        use_dictionary=True,
    )

    try:
        for file_index, input_part in enumerate(
            input_parts,
            start=1,
        ):
            temp_path = temp_paths[input_part.name]

            print(
                f"[merge {file_index}/{len(input_parts)}] "
                f"{input_part.name}",
                flush=True,
            )

            with pa.memory_map(
                str(temp_path),
                "r",
            ) as source:
                reader = ipc.open_file(source)

                for batch_index in range(
                    reader.num_record_batches
                ):
                    batch = reader.get_batch(
                        batch_index
                    )

                    writer.write_table(
                        pa.Table.from_batches(
                            [batch],
                            schema=OUTPUT_SCHEMA,
                        ),
                        row_group_size=ROW_GROUP_SIZE,
                    )
    finally:
        writer.close()


# ============================================================================
# Preview helpers
# ============================================================================

def print_mapping_preview(
    input_part: Path,
) -> None:
    print("")
    print(
        f"Mapping/time preview from {input_part.name}:"
    )

    printed = 0

    with input_part.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as stream:
        for line in stream:
            cols = line.rstrip("\r\n").split("\t", 6)

            if len(cols) < 6:
                continue

            adidxs = cols[2].strip().split(",")
            times = cols[4].strip().split(",")
            ctypes = cols[5].strip().split(",")

            n = min(
                len(adidxs),
                len(times),
                len(ctypes),
            )

            if n <= 0:
                continue

            try:
                anchor = parse_timestamp_seconds(
                    times[n - 1]
                )
            except Exception:
                continue

            for i in range(
                max(0, n - 10),
                n,
            ):
                adidx = adidxs[i].strip()
                if not adidx:
                    continue

                try:
                    hist_time = parse_timestamp_seconds(
                        times[i]
                    )
                    preview_stats = Counter()
                    bucket = time_bucket(
                        hist_time,
                        anchor,
                        preview_stats,
                    )
                except Exception:
                    bucket = None

                print(
                    f"  adidx={adidx} "
                    f"ctype={ctypes[i].strip()} "
                    f"sid_hit={adidx in _ADIDX2SID_SUFFIX} "
                    f"time_bucket={bucket}"
                )

                printed += 1

                if printed >= 10:
                    return


def print_first_output_row() -> None:
    parquet_file = pq.ParquetFile(
        OUTPUT_PARQUET
    )

    if parquet_file.metadata.num_rows <= 0:
        print("Output parquet is empty.")
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
        .slice(0, 1)
    )

    row = table.to_pylist()[0]

    print("")
    print("First row:")
    print(f"source   : {row['source']}")
    print(f"uuid     : {row['uuid']}")
    print(f"messages : {row['messages']}")
    print(f"metadata : {row['metadata']}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    global _ADIDX2SID_SUFFIX
    global _MID2TOKEN

    check_paths()

    input_parts = discover_input_parts()

    print(f"Input dir      : {INPUT_DIR}")
    print(f"Found parts    : {len(input_parts):,}")
    print(f"Output         : {OUTPUT_PARQUET}")
    print(f"Workers        : {NUM_WORKERS}")
    print(f"Batch size     : {WRITE_BATCH_SIZE:,}")
    print(f"History max    : {HIST_MAX_LEN}")
    print(f"Time buckets   : 0..{MAX_TIME_BUCKET}")
    print(
        "Time anchor    : "
        + (
            "dropped last target timestamp"
            if DROP_LAST_ITEM_AS_TARGET
            else "latest observed history timestamp"
        )
    )

    overall_start = time.perf_counter()

    # 1) item mapping
    print("")
    print("Loading adid -> SID suffix...")

    adid2sid_suffix = load_adid2sid_suffix()

    print(
        f"Loaded adid SID: {len(adid2sid_suffix):,}"
    )

    print("Building adidx -> SID suffix...")

    (
        _ADIDX2SID_SUFFIX,
        mapping_stats,
    ) = load_adidx2sid_suffix(
        adid2sid_suffix
    )

    del adid2sid_suffix

    print(
        f"Built adidx SID: {len(_ADIDX2SID_SUFFIX):,}"
    )
    print(
        f"adidx rows      : "
        f"{mapping_stats['adidx2adid_rows']:,}"
    )
    print(
        f"adid no SID     : "
        f"{mapping_stats['adid_without_sid']:,}"
    )

    # 2) MID mapping
    print("Loading mid -> token...")

    _MID2TOKEN = load_mid2token()

    print(
        f"Loaded MID token: {len(_MID2TOKEN):,}"
    )

    print_mapping_preview(
        input_parts[0]
    )

    # 3) temp paths
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    TEMP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_paths: Dict[str, Path] = {
        input_part.name: (
            TEMP_DIR
            / (input_part.name + ".arrow")
        )
        for input_part in input_parts
    }

    # 4) parallel conversion
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

    # Linux fork shares the large read-only mappings by COW.
    ctx = mp.get_context("fork")

    results: Dict[str, Dict[str, Any]] = {}

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
                    str(temp_paths[input_part.name]),
                )
                future_to_part[future] = input_part

            completed = 0

            for future in as_completed(
                future_to_part
            ):
                input_part = future_to_part[
                    future
                ]

                result = future.result()

                results[
                    input_part.name
                ] = result

                completed += 1

                print(
                    f"[done {completed}/{len(input_parts)}] "
                    f"{input_part.name}: "
                    f"input={result['input_rows']:,}, "
                    f"output={result['output_rows']:,}, "
                    f"skipped={result['skipped_rows']:,}, "
                    f"time={result['elapsed']:.2f}s",
                    flush=True,
                )

        # 5) merge
        print("")
        print(
            "Merging temporary Arrow files "
            "into one Parquet..."
        )

        merge_start = time.perf_counter()

        merge_temp_arrow_files(
            input_parts=input_parts,
            temp_paths=temp_paths,
        )

        merge_elapsed = (
            time.perf_counter()
            - merge_start
        )

        # 6) stats
        total_input = sum(
            int(result["input_rows"])
            for result in results.values()
        )
        total_output = sum(
            int(result["output_rows"])
            for result in results.values()
        )
        total_skipped = sum(
            int(result["skipped_rows"])
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
            total_input / overall_elapsed
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
            f"TIME tokens          : "
            f"{total_stats['time_token_items']:,}"
        )
        print(
            f"Bad time items       : "
            f"{total_stats['bad_time_items']:,}"
        )
        print(
            f"Bad anchor rows      : "
            f"{total_stats['bad_anchor_time_rows']:,}"
        )
        print(
            f"Negative time delta  : "
            f"{total_stats['negative_time_delta_items']:,}"
        )
        print(
            f"TIME clamped to 336  : "
            f"{total_stats['time_bucket_clamped_items']:,}"
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
