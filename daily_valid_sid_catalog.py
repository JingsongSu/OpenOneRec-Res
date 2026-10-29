#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a CONSISTENT daily item -> SID snapshot and GPU Trie before inference.

Source (updates daily):
  /home/jovyan/zhouyuhang-cloud1/sujinsong/rq_daily/sid_current/ad_sid.parquet

The writer MUST read the returned canonical snapshot, NOT the mutable source.
The vLLM pooler MUST use the index returned by the SAME build call.
This tool rebuilds for every inference launch, which safely picks up a new
catalog after the upstream daily job has published the Parquet.

Automatic housekeeping: AFTER a successful new build, remove previous
catalog directories older than the configured retention period (default
3 days), but only if their lifetime lock is no longer held. The current
inference parent keeps that lock open until it exits; concurrent active
jobs cannot lose their catalog. Pre-lock legacy directories are left
untouched for safety. The source Parquet and recommendation output files
are never deleted.

Accepted input columns (case insensitive):
  ad (preferred), adid/ad_id/item_id/itemid; sid (4 integer local codes)
Accepted SID values: int[4], JSON array, "(1,2,3,4)", "1,2,3,4",
or four <s_a_...><s_b_...><s_c_...><s_d_...> tokens.
All SID IDs must be local per-layer IDs, exactly as used by the online writer.

A row with any SID code outside the model's legal [0, size) interval is
considered NOT a valid item->SID mapping. It is skipped from BOTH the
snapshot consumed by the online writer AND the constrained-decode index.
Malformed SID syntax and conflicting mappings remain hard errors.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, BinaryIO, Dict, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_SOURCE = Path(
    "/home/jovyan/zhouyuhang-cloud1/sujinsong/rq_daily/"
    "sid_current/ad_sid.parquet"
)
ADID_COLUMNS = ("ad", "adid", "ad_id", "item_id", "itemid")
SID_COLUMNS = ("sid", "ad_sid", "semid", "semantic_id", "sid_code")
SID_TOKENS = re.compile(r"<s_([abcd])_(\d+)>")
BAD_STRINGS = {"", "none", "null", "nan", "<na>"}


# Keep three days by default, configurable WITHOUT changing your inference
# scripts: DAILY_SID_CATALOG_KEEP_DAYS=3
DEFAULT_CATALOG_KEEP_DAYS = 3
CATALOG_LOCK_FILE = ".catalog.in_use.lock"
_RUN_DIRECTORY_PATTERN = re.compile(r"^\d{8}T\d{6}_\d+_[0-9a-f]{8}$")

# Keep the OS-level flock open for the lifetime of the parent inference Python
# process. This module is imported into the parent BEFORE its GPU/writer spawn.
# The lock is automatically released by the OS when the parent exits.
_ACTIVE_RUN_LOCKS: Dict[str, BinaryIO] = {}


def _catalog_keep_days() -> int:
    value = os.environ.get(
        "DAILY_SID_CATALOG_KEEP_DAYS",
        str(DEFAULT_CATALOG_KEEP_DAYS),
    )
    try:
        days = int(value)
    except ValueError as exc:
        raise ValueError(
            f"DAILY_SID_CATALOG_KEEP_DAYS must be a positive integer; "
            f"received {value!r}"
        ) from exc
    if days < 1:
        raise ValueError(
            f"DAILY_SID_CATALOG_KEEP_DAYS must be >= 1, got {days}"
        )
    return days


def _acquire_run_lock(run_dir: Path) -> None:
    """Register a POSIX flock held until this parent process exits."""
    lock_file = (run_dir / CATALOG_LOCK_FILE).open("a+b")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        lock_file.close()
        raise
    _ACTIVE_RUN_LOCKS[str(run_dir.resolve())] = lock_file


def _release_run_lock(run_dir: Path) -> None:
    """Only used on build FAILURE; successful runs retain the parent lock."""
    lock_file = _ACTIVE_RUN_LOCKS.pop(str(run_dir.resolve()), None)
    if lock_file is not None:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()


def cleanup_old_catalog_runs(
    output_root: Path,
    current_run_dir: Path,
    keep_days: int,
) -> Dict[str, int]:
    """Remove ONLY expired, unlocked, recognized generated run directories.

    Never delete arbitrary files, unrelated directories, or symlinks. When
    a previous inference parent is still alive, LOCK_NB fails and that
    directory is always kept. Legacy directories without our lock file
    are deliberately skipped because their active state cannot be proven.
    """
    output_root = Path(output_root)
    current_run_dir = Path(current_run_dir)
    if keep_days < 1:
        raise ValueError("keep_days must be >= 1")

    cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
    stats = {
        "deleted": 0,
        "active_skipped": 0,
        "legacy_skipped": 0,
        "other_skipped": 0,
        "delete_failed": 0,
    }
    if not output_root.is_dir():
        return stats

    for run_dir in output_root.iterdir():
        # Never follow symlinks; never touch files or foreign directories.
        if run_dir.is_symlink() or not run_dir.is_dir():
            continue
        if run_dir == current_run_dir:
            continue
        if not _RUN_DIRECTORY_PATTERN.fullmatch(run_dir.name):
            continue
        try:
            started = datetime.strptime(
                run_dir.name.split("_", 1)[0], "%Y%m%dT%H%M%S"
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            stats["other_skipped"] += 1
            continue
        if started >= cutoff:
            continue

        lock_path = run_dir / CATALOG_LOCK_FILE
        if lock_path.is_symlink():
            stats["other_skipped"] += 1
            continue
        if not lock_path.is_file():
            # Older versions do not have a lifetime lock: they may still
            # be running, so do NOT assume that they are safe to delete.
            stats["legacy_skipped"] += 1
            continue

        try:
            lock_file = lock_path.open("a+b")
        except OSError as exc:
            stats["delete_failed"] += 1
            print(
                f"[DailySID] cleanup: cannot open lock for {run_dir}: {exc}",
                flush=True,
            )
            continue

        try:
            try:
                fcntl.flock(
                    lock_file.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except BlockingIOError:
                stats["active_skipped"] += 1
                continue

            # The lock guarantees no participating parent can be using
            # this run. Recheck pathname before recursive deletion.
            if not run_dir.is_dir() or run_dir.is_symlink():
                stats["other_skipped"] += 1
                continue
            try:
                shutil.rmtree(run_dir)
                stats["deleted"] += 1
            except OSError as exc:
                stats["delete_failed"] += 1
                print(
                    f"[DailySID] cleanup: could not remove {run_dir}: {exc}",
                    flush=True,
                )
        finally:
            lock_file.close()

    print(
        "[DailySID] cleanup finished: "
        f"keep_days={keep_days}, "
        f"deleted={stats['deleted']}, "
        f"active_protected={stats['active_skipped']}, "
        f"legacy_unlocked_skipped={stats['legacy_skipped']}, "
        f"delete_failed={stats['delete_failed']}",
        flush=True,
    )
    return stats



def adid_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        raise ValueError(
            "ad must be a string or integer (e.g. uint64). Float64 may "
            "lose precision for the 20-digit ad IDs in this catalog."
        )
    result = str(value).strip()
    return "" if result.lower() in BAD_STRINGS else result


def sid_to_local_tuple(value: Any) -> Tuple[int, int, int, int]:
    if value is None:
        raise ValueError("SID is null")
    if isinstance(value, np.ndarray):
        parts = value.tolist()
    elif isinstance(value, (tuple, list)):
        parts = list(value)
    elif isinstance(value, str):
        text = value.strip()
        tokens = SID_TOKENS.findall(text)
        if len(tokens) == 4 and [x[0] for x in tokens] == list("abcd"):
            parts = [x[1] for x in tokens]
        else:
            # JSON-style and Python tuple/string representations
            stripped = text.strip("[]()").replace("'", "").replace('"', "")
            parts = [p for p in re.split(r"[\s,]+", stripped) if p]
    else:
        raise ValueError(f"Unsupported SID data type: {type(value)}")

    if len(parts) != 4:
        raise ValueError(f"Expected 4 local SID codes, got {value!r}")

    parsed = []
    for layer, x in zip("abcd", parts):
        if isinstance(x, str):
            m = re.fullmatch(r"<s_([abcd])_(\d+)>", x.strip())
            if m:
                if m.group(1) != layer:
                    raise ValueError(
                        f"SID token {x!r} appears in wrong layer {layer}"
                    )
                x = m.group(2)
        if isinstance(x, (float, np.floating)):
            if not math.isfinite(x) or not float(x).is_integer():
                raise ValueError(f"Non-integer SID component {x!r}")
            parsed.append(int(x))
        else:
            parsed.append(int(x))
    return tuple(parsed)  # type: ignore[return-value]


def choose_columns(names):
    lookup = {col.lower(): col for col in names}
    adid = next((lookup[s] for s in ADID_COLUMNS if s in lookup), None)
    sid = next((lookup[s] for s in SID_COLUMNS if s in lookup), None)
    if not adid or not sid:
        raise ValueError(
            "Could not identify columns for valid mapping; "
            f"available columns={names}. Expected ad and sid (or supported aliases)."
        )
    return adid, sid


def build_daily_sid_catalog(
    source_path: Path,
    exported_config_path: Path,
    output_root: Path,
    beam_size: int = 50,
) -> Dict[str, Any]:
    """Build once at launcher startup; return immutable snapshot/index paths."""
    source_path = Path(source_path).resolve(strict=True)
    exported_config_path = Path(exported_config_path).resolve(strict=True)
    output_root = Path(output_root)
    keep_days = _catalog_keep_days()
    with exported_config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    sizes = tuple(int(s) for s in config["residual_sid_layer_sizes"])
    if len(sizes) != 4 or any(s <= 0 for s in sizes):
        raise ValueError(f"Four positive SID layer sizes required, got {sizes}")
    configured_beam = int(config["residual_sid_beam_size"])
    if configured_beam != int(beam_size):
        raise ValueError(
            f"Beam mismatch: exported={configured_beam}, launcher={beam_size}"
        )

    # Packed 4-layer local SID must fit into signed int64 for GPU searchsorted.
    capacity = math.prod(sizes)
    if capacity > np.iinfo(np.int64).max:
        raise ValueError(f"Packed SID key capacity overflows int64: {capacity}")

    source_stat_before = source_path.stat()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + f"_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    )
    run_dir = output_root / run_id
    run_dir.mkdir(mode=0o750, exist_ok=False)
    # Lock immediately after creation, including the expensive snapshot build.
    # Old runs cannot be reclaimed while their parent is still alive.
    try:
        _acquire_run_lock(run_dir)
    except BaseException:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    snapshot = run_dir / "ad_sid_snapshot.parquet"
    index_path = run_dir / "valid_sid_trie.npz"

    # Normalize to the schema expected by the EXISTING online writer.
    schema = pa.schema([("adid", pa.string()), ("sid", pa.string())])
    mappings = {}  # packed valid SID -> adid (detect ambiguous reverse maps)
    duplicate_same = 0
    skipped_blank_adid = 0
    skipped_out_of_range = 0
    skipped_out_of_range_by_layer = [0, 0, 0, 0]
    out_of_range_examples = []  # at most 5 examples for diagnostics
    accepted_rows = 0
    total_rows = 0
    digest = hashlib.sha256()
    writer = None

    try:
        parquet = pq.ParquetFile(str(source_path))
        adid_col, sid_col = choose_columns(parquet.schema_arrow.names)
        adid_type = parquet.schema_arrow.field(adid_col).type
        if pa.types.is_floating(adid_type):
            raise ValueError(
                f"{adid_col!r} column is {adid_type}: float storage is "
                "unsafe for 20-digit advertisement IDs. Export as "
                "uint64, decimal128 or string to preserve precision."
            )

        writer = pq.ParquetWriter(str(snapshot), schema, compression="zstd")
        for batch in parquet.iter_batches(
            batch_size=100_000, columns=[adid_col, sid_col], use_threads=False
        ):
            # Resolve by actual RecordBatch field names, not physical column
            # order (some daily jobs write sid before ad).
            ads = batch.column(
                batch.schema.get_field_index(adid_col)
            ).to_pylist()
            sids = batch.column(
                batch.schema.get_field_index(sid_col)
            ).to_pylist()
            normalized_ids = []
            normalized_sids = []
            for ad, sid_raw in zip(ads, sids):
                total_rows += 1
                adid = adid_to_text(ad)
                if not adid:
                    skipped_blank_adid += 1
                    continue

                try:
                    local = sid_to_local_tuple(sid_raw)
                except Exception as exc:
                    raise ValueError(
                        f"Invalid daily SID on row {total_rows}: "
                        f"adid={adid!r}, sid={sid_raw!r}"
                    ) from exc
                # Skip invalid/out-of-model codes; DO NOT put them in
                # either the Trie index or the writer's SID->ad snapshot.
                # For example, if D-layer size=15000, D=15000 is invalid:
                # valid D codes are 0..14999.
                bad_layers = [
                    layer
                    for layer, (code, size) in enumerate(zip(local, sizes))
                    if code < 0 or code >= size
                ]
                if bad_layers:
                    skipped_out_of_range += 1
                    for layer in bad_layers:
                        skipped_out_of_range_by_layer[layer] += 1
                    if len(out_of_range_examples) < 5:
                        out_of_range_examples.append(
                            {
                                "row": total_rows,
                                "adid": adid,
                                "sid": list(local),
                                "invalid_layers": bad_layers,
                            }
                        )
                    continue

                accepted_rows += 1

                key = 0
                for code, size in zip(local, sizes):
                    key = key * size + code
                previous = mappings.get(key)
                if previous is not None:
                    if previous != adid:
                        raise ValueError(
                            f"SID {local} maps to TWO different items: "
                            f"{previous!r} and {adid!r}. The original online "
                            "writer also rejects this; fix daily mapping."
                        )
                    duplicate_same += 1
                else:
                    mappings[key] = adid

                sid_text = json.dumps(local, separators=(",", ":"))
                normalized_ids.append(adid)
                normalized_sids.append(sid_text)
                digest.update(adid.encode("utf-8"))
                digest.update(b"\t")
                digest.update(sid_text.encode("ascii"))
                digest.update(b"\n")

            if normalized_ids:
                writer.write_table(
                    pa.table(
                        {
                            "adid": pa.array(normalized_ids, type=pa.string()),
                            "sid": pa.array(normalized_sids, type=pa.string()),
                        },
                        schema=schema,
                    )
                )
        writer.close()
        writer = None

        source_stat_after = source_path.stat()
        source_signature_before = (
            source_stat_before.st_size,
            source_stat_before.st_mtime_ns,
            source_stat_before.st_ino,
        )
        source_signature_after = (
            source_stat_after.st_size,
            source_stat_after.st_mtime_ns,
            source_stat_after.st_ino,
        )
        if source_signature_before != source_signature_after:
            raise RuntimeError(
                "Daily Parquet changed during snapshot build. "
                "Retry after upstream publication finishes."
            )

        if len(mappings) < int(beam_size):
            raise ValueError(
                f"Daily catalog only contains {len(mappings)} distinct SID; "
                f"cannot produce {beam_size} distinct valid SID"
            )

        k4 = np.asarray(sorted(mappings), dtype=np.int64)
        k3 = np.unique(k4 // sizes[3])
        k2 = np.unique(k3 // sizes[2])
        k1 = np.unique(k2 // sizes[1])
        with index_path.open("wb") as f:
            np.savez_compressed(
                f,
                sizes=np.asarray(sizes, dtype=np.int64),
                k1=k1,
                k2=k2,
                k3=k3,
                k4=k4,
                index_version=np.asarray([1], dtype=np.int64),
            )
        metadata = {
            "source": str(source_path),
            "source_mtime_ns": source_stat_before.st_mtime_ns,
            "source_size_bytes": source_stat_before.st_size,
            "source_columns": {"adid": adid_col, "sid": sid_col},
            "snapshot": str(snapshot.resolve()),
            "index": str(index_path.resolve()),
            "snapshot_sha256_rows": digest.hexdigest(),
            "source_row_count": total_rows,
            "accepted_rows": accepted_rows,
            "skipped_blank_adid": skipped_blank_adid,
            "skipped_out_of_range": skipped_out_of_range,
            "skipped_out_of_range_by_layer": {
                name: skipped_out_of_range_by_layer[i]
                for i, name in enumerate(("A", "B", "C", "D"))
            },
            "out_of_range_examples": out_of_range_examples,
            "duplicate_same_adid": duplicate_same,
            "unique_valid_sid": len(mappings),
            "prefix_counts": [len(k1), len(k2), len(k3), len(k4)],
            "layer_sizes": list(sizes),
            "beam_size": int(beam_size),
            "built_utc": datetime.now(timezone.utc).isoformat(),
            "catalog_keep_days": keep_days,
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # Printed here (not just in main) so it also appears in the
        # full online inference script's startup log.
        print(
            "[DailySID] catalog built: "
            f"source_rows={total_rows:,}, "
            f"accepted_rows={accepted_rows:,}, "
            f"out_of_range_skipped={skipped_out_of_range:,}, "
            f"blank_ad_skipped={skipped_blank_adid:,}, "
            f"unique_valid_sid={len(mappings):,}",
            flush=True,
        )
        print(
            "[DailySID] out-of-range rows by layer: "
            + ", ".join(
                f"{name}={skipped_out_of_range_by_layer[i]:,}"
                for i, name in enumerate(("A", "B", "C", "D"))
            ),
            flush=True,
        )
        if out_of_range_examples:
            print(
                "[DailySID] first excluded examples: "
                + json.dumps(
                    out_of_range_examples, ensure_ascii=False
                ),
                flush=True,
            )
        # Housekeeping is deliberately AFTER a fully successful build.
        # A transient cleanup failure cannot invalidate this new catalog.
        try:
            cleanup_old_catalog_runs(
                output_root=output_root,
                current_run_dir=run_dir,
                keep_days=keep_days,
            )
        except Exception as exc:
            print(
                f"[DailySID] WARNING: catalog built successfully but "
                f"old-catalog cleanup failed: {exc!r}",
                flush=True,
            )
        return metadata
    except Exception:
        if writer is not None:
            writer.close()
        try:
            shutil.rmtree(run_dir, ignore_errors=True)
        finally:
            _release_run_lock(run_dir)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--beam-size", type=int, default=50)
    args = parser.parse_args()
    result = build_daily_sid_catalog(
        args.source, args.model_config, args.output_root, args.beam_size
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
