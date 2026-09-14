#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filter inference adid candidates by a legal adid whitelist.

Python compatibility:
    Python 3.6+

Input inference format:
    mid_ctype<TAB>adid1<TAB>adid2<TAB>...

Legal adid file:
    one legal adid per line

Output:
    - Keep the first column (mid_ctype) unchanged.
    - Filter columns 2..N by legal adid whitelist.
    - Preserve candidate order.
    - Drop the entire row if no candidate adid remains.
    - Preserve each input part-* filename.
    - Write filter_stats.json into output directory.

Example:
    python3 filter_infer_adids.py \
        --input-dir /path/to/infer_adid_parts \
        --legal-adids /path/to/legal_adids.txt \
        --output-dir /path/to/infer_adid_parts_filtered.tmp.xxx
"""

import argparse
import glob
import json
import os
import sys
import time


def log(message):
    sys.stdout.write(str(message) + "\n")
    sys.stdout.flush()


def load_legal_adids(path):
    """
    Load legal adids as bytes.

    Using bytes avoids unnecessary decode/encode work while scanning large
    inference files and keeps exact byte matching semantics.
    """
    legal_adids = set()
    total_lines = 0
    empty_lines = 0

    with open(path, "rb") as f:
        for raw_line in f:
            total_lines += 1
            adid = raw_line.strip()

            if not adid:
                empty_lines += 1
                continue

            # Defensive handling:
            # legal_adids.txt should normally contain one adid per line.
            # If a tab/space accidentally exists, use only the first field.
            if b"\t" in adid:
                adid = adid.split(b"\t", 1)[0].strip()
            elif b" " in adid:
                adid = adid.split(None, 1)[0].strip()

            if adid:
                legal_adids.add(adid)

    return legal_adids, total_lines, empty_lines


def list_part_files(input_dir):
    """
    Return sorted part-* regular files.
    Ignore markers such as _SUCCESS and other metadata files.
    """
    pattern = os.path.join(input_dir, "part-*")
    files = []

    for path in glob.glob(pattern):
        if os.path.isfile(path):
            files.append(path)

    files.sort()
    return files


def filter_one_part(input_path, output_path, legal_adids):
    stats = {
        "input_rows": 0,
        "output_rows": 0,
        "dropped_rows": 0,
        "malformed_rows": 0,
        "input_candidate_adids": 0,
        "kept_candidate_adids": 0,
        "removed_candidate_adids": 0,
    }

    with open(input_path, "rb") as src, open(output_path, "wb") as dst:
        for raw_line in src:
            stats["input_rows"] += 1

            line = raw_line.rstrip(b"\r\n")
            if not line:
                stats["malformed_rows"] += 1
                stats["dropped_rows"] += 1
                continue

            fields = line.split(b"\t")

            # Inference format must contain:
            #   first column = mid_ctype
            #   remaining columns = candidate adids
            if len(fields) < 2:
                stats["malformed_rows"] += 1
                stats["dropped_rows"] += 1
                continue

            key = fields[0]
            candidates = fields[1:]

            stats["input_candidate_adids"] += len(candidates)

            kept = []
            for adid in candidates:
                adid = adid.strip()
                if adid and adid in legal_adids:
                    kept.append(adid)

            kept_count = len(kept)
            removed_count = len(candidates) - kept_count

            stats["kept_candidate_adids"] += kept_count
            stats["removed_candidate_adids"] += removed_count

            if not kept:
                stats["dropped_rows"] += 1
                continue

            dst.write(key)
            dst.write(b"\t")
            dst.write(b"\t".join(kept))
            dst.write(b"\n")
            stats["output_rows"] += 1

    return stats


def add_stats(total, part_stats):
    for key in (
        "input_rows",
        "output_rows",
        "dropped_rows",
        "malformed_rows",
        "input_candidate_adids",
        "kept_candidate_adids",
        "removed_candidate_adids",
    ):
        total[key] += part_stats[key]


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter inference candidate adids by legal adid whitelist."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing original inference part-* files.",
    )
    parser.add_argument(
        "--legal-adids",
        required=True,
        help="Whitelist file containing one legal adid per line.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where filtered part-* files will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    input_dir = os.path.abspath(args.input_dir)
    legal_adids_path = os.path.abspath(args.legal_adids)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(input_dir):
        log("ERROR: input directory does not exist: {0}".format(input_dir))
        return 2

    if not os.path.isfile(legal_adids_path):
        log("ERROR: legal adid file does not exist: {0}".format(legal_adids_path))
        return 2

    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        log("ERROR: output path exists but is not a directory: {0}".format(output_dir))
        return 2

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    part_files = list_part_files(input_dir)
    if not part_files:
        log("ERROR: no part-* files found in: {0}".format(input_dir))
        return 3

    log("loading legal adid whitelist...")
    legal_adids, legal_file_lines, legal_empty_lines = load_legal_adids(
        legal_adids_path
    )

    if not legal_adids:
        log("ERROR: legal adid whitelist is empty: {0}".format(legal_adids_path))
        return 4

    log("legal adid unique count: {0}".format(len(legal_adids)))
    log("input part count: {0}".format(len(part_files)))

    total = {
        "input_rows": 0,
        "output_rows": 0,
        "dropped_rows": 0,
        "malformed_rows": 0,
        "input_candidate_adids": 0,
        "kept_candidate_adids": 0,
        "removed_candidate_adids": 0,
    }

    part_stats_list = []

    for index, input_path in enumerate(part_files, 1):
        part_name = os.path.basename(input_path)
        output_path = os.path.join(output_dir, part_name)

        log(
            "[{0}/{1}] filtering {2}".format(
                index,
                len(part_files),
                part_name,
            )
        )

        part_stats = filter_one_part(
            input_path=input_path,
            output_path=output_path,
            legal_adids=legal_adids,
        )

        add_stats(total, part_stats)

        part_record = {"part": part_name}
        part_record.update(part_stats)
        part_stats_list.append(part_record)

    elapsed_seconds = time.time() - start_time

    stats = {
        "input_dir": input_dir,
        "legal_adids_file": legal_adids_path,
        "output_dir": output_dir,
        "input_part_count": len(part_files),
        "output_part_count": len(part_files),
        "legal_file_lines": legal_file_lines,
        "legal_file_empty_lines": legal_empty_lines,
        "legal_unique_adids": len(legal_adids),
        "input_rows": total["input_rows"],
        "output_rows": total["output_rows"],
        "dropped_rows": total["dropped_rows"],
        "malformed_rows": total["malformed_rows"],
        "input_candidate_adids": total["input_candidate_adids"],
        "kept_candidate_adids": total["kept_candidate_adids"],
        "removed_candidate_adids": total["removed_candidate_adids"],
        "row_keep_ratio": safe_ratio(
            total["output_rows"],
            total["input_rows"],
        ),
        "candidate_keep_ratio": safe_ratio(
            total["kept_candidate_adids"],
            total["input_candidate_adids"],
        ),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "parts": part_stats_list,
    }

    stats_path = os.path.join(output_dir, "filter_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    log("============================================================")
    log("filter finished successfully")
    log("input parts          : {0}".format(len(part_files)))
    log("legal unique adids   : {0}".format(len(legal_adids)))
    log("input rows           : {0}".format(total["input_rows"]))
    log("output rows          : {0}".format(total["output_rows"]))
    log("dropped rows         : {0}".format(total["dropped_rows"]))
    log("malformed rows       : {0}".format(total["malformed_rows"]))
    log(
        "input candidate adids: {0}".format(
            total["input_candidate_adids"]
        )
    )
    log(
        "kept candidate adids : {0}".format(
            total["kept_candidate_adids"]
        )
    )
    log(
        "removed candidate ids: {0}".format(
            total["removed_candidate_adids"]
        )
    )
    log(
        "candidate keep ratio : {0:.6f}".format(
            stats["candidate_keep_ratio"]
        )
    )
    log("stats file           : {0}".format(stats_path))
    log("elapsed seconds       : {0}".format(stats["elapsed_seconds"]))
    log("============================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
