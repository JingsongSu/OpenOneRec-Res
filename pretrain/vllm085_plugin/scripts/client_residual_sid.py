from __future__ import annotations

import argparse
import json

import requests
from transformers import AutoConfig, AutoTokenizer

from openonerec_vllm085_residual_sid.codec import (
    candidates_to_records,
    unpack_data,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--served_model", default="onerec-residual-sid")
    parser.add_argument(
        "--model_config",
        required=True,
        help="Local vLLM-exported model directory.",
    )
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(
        args.model_config,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_config,
        trust_remote_code=True,
    )

    response = requests.post(
        args.url.rstrip("/") + "/pooling",
        json={
            "model": args.served_model,
            "input": [args.prompt],
        },
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()
    packed = payload["data"][0]["data"]

    candidates = unpack_data(
        packed,
        beam_size=int(config.residual_sid_beam_size),
        num_layers=4,
    )
    records = candidates_to_records(
        candidates,
        config.residual_sid_layer_starts,
    )
    for record in records:
        record["tokens"] = tokenizer.convert_ids_to_tokens(
            record["global_ids"]
        )
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
