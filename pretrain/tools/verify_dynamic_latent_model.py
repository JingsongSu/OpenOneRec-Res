#!/usr/bin/env python3
"""Verify a converted/exported branch-conditioned interleaved latent residual SID model directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Set

from transformers import AutoTokenizer


def collect_weight_keys(model_dir: Path) -> Set[str]:
    index_path = model_dir / "model.safetensors.index.json"

    if index_path.is_file():
        index = json.loads(
            index_path.read_text(
                encoding="utf-8"
            )
        )
        return set(
            index.get(
                "weight_map",
                {},
            ).keys()
        )

    safetensor_files = sorted(
        model_dir.glob("*.safetensors")
    )

    if not safetensor_files:
        raise FileNotFoundError(
            "No model.safetensors.index.json "
            "or *.safetensors files found in "
            f"{model_dir}"
        )

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is required to inspect "
            "an unindexed checkpoint."
        ) from exc

    keys: Set[str] = set()

    for path in safetensor_files:
        with safe_open(
            str(path),
            framework="pt",
            device="cpu",
        ) as handle:
            keys.update(
                handle.keys()
            )

    return keys


def require_keys(
    keys: Set[str],
    expected: Iterable[str],
    description: str,
) -> None:
    missing = [
        key
        for key in expected
        if key not in keys
    ]

    if missing:
        raise ValueError(
            f"Missing {description} weights: "
            f"{missing}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
    )
    parser.add_argument(
        "--expect_vllm_export",
        action="store_true",
    )
    args = parser.parse_args()

    model_dir = Path(
        args.model
    ).resolve()

    config_path = (
        model_dir / "config.json"
    )

    if not config_path.is_file():
        raise FileNotFoundError(
            config_path
        )

    config = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    print("model =", model_dir)
    print(
        "architectures =",
        config.get(
            "architectures"
        ),
    )
    print(
        "latent_reasoning_enabled =",
        config.get(
            "latent_reasoning_enabled"
        ),
    )
    print(
        "latent_reasoning_mode =",
        config.get(
            "latent_reasoning_mode"
        ),
    )
    print(
        "latent_reasoning_num_steps =",
        config.get(
            "latent_reasoning_num_steps"
        ),
    )
    print(
        "latent_reasoning_num_transitions =",
        config.get("latent_reasoning_num_transitions"),
    )
    print(
        "latent_reasoning_conditioning =",
        config.get("latent_reasoning_conditioning"),
    )
    print(
        "latent_reasoning_update =",
        config.get("latent_reasoning_update"),
    )
    print(
        "latent_reasoning_loss_weight =",
        config.get(
            "latent_reasoning_loss_weight"
        ),
    )

    if not bool(
        config.get(
            "latent_reasoning_enabled",
            False,
        )
    ):
        raise ValueError(
            "latent_reasoning_enabled is False/missing."
        )

    if (
        config.get(
            "latent_reasoning_mode"
        )
        != "branch_conditioned_interleaved"
    ):
        raise ValueError(
            "Wrong latent_reasoning_mode."
        )

    num_steps = int(
        config.get(
            "latent_reasoning_num_steps",
            0,
        )
    )

    num_sid_layers = len(
        config.get("residual_sid_layer_starts", [])
    )
    expected_steps = num_sid_layers - 1
    if num_steps != expected_steps:
        raise ValueError(
            "Expected one interleaved latent step before B/C/D: "
            f"steps={num_steps}, expected={expected_steps}."
        )
    num_transitions = int(
        config.get(
            "latent_reasoning_num_transitions",
            0,
        )
    )
    if num_transitions != expected_steps:
        raise ValueError(
            "Expected exactly sid_layers-1 latent transitions: "
            f"transitions={num_transitions}, expected={expected_steps}."
        )
    if config.get("latent_reasoning_conditioning") != "hard_previous_sid":
        raise ValueError(
            "latent_reasoning_conditioning must be 'hard_previous_sid'."
        )
    if config.get("latent_reasoning_update") != "thought_then_formal_residual":
        raise ValueError(
            "latent_reasoning_update must be "
            "'thought_then_formal_residual'."
        )
    if float(config.get("latent_reasoning_loss_weight", 0.0)) != 0.0:
        raise ValueError(
            "Branch-conditioned interleaved reasoning uses no auxiliary latent CE."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
    )

    print(
        "chat_template exists =",
        tokenizer.chat_template is not None,
    )

    sid_begin_id = (
        tokenizer.convert_tokens_to_ids(
            "<|sid_begin|>"
        )
    )

    config_sid_begin_id = int(
        config[
            "residual_sid_begin_token_id"
        ]
    )

    print(
        "sid_begin tokenizer/config =",
        sid_begin_id,
        config_sid_begin_id,
    )

    if (
        int(sid_begin_id)
        != config_sid_begin_id
    ):
        raise ValueError(
            "SID_BEGIN tokenizer/config mismatch."
        )

    keys = collect_weight_keys(
        model_dir
    )

    latent_expected = []

    for step in range(
        num_transitions
    ):
        prefix = (
            f"latent_reasoning_blocks."
            f"{step}."
        )
        latent_expected.extend(
            [
                prefix + "linear.weight",
                prefix + "linear.bias",
                prefix + "layer_norm.weight",
                prefix + "layer_norm.bias",
            ]
        )

    require_keys(
        keys,
        latent_expected,
        "layer-wise latent transition",
    )

    unexpected_prefix = (
        f"latent_reasoning_blocks.{num_transitions}."
    )
    unexpected = sorted(
        key for key in keys
        if key.startswith(unexpected_prefix)
    )
    if unexpected:
        raise ValueError(
            "Found an extra latent transition block; expected exactly "
            f"{num_transitions}: {unexpected[:8]}"
        )

    residual_expected = []

    for step in range(
        max(
            0,
            num_sid_layers - 1,
        )
    ):
        prefix = (
            f"sid_residual_blocks."
            f"{step}."
        )
        residual_expected.extend(
            [
                prefix + "linear.weight",
                prefix + "linear.bias",
                prefix + "layer_norm.weight",
                prefix + "layer_norm.bias",
            ]
        )

    require_keys(
        keys,
        residual_expected,
        "residual SID",
    )

    if args.expect_vllm_export:
        architectures = list(
            config.get(
                "architectures",
                [],
            )
        )

        if (
            "Qwen3ForResidualSIDPoolingV085"
            not in architectures
        ):
            raise ValueError(
                "Expected vLLM architecture "
                "Qwen3ForResidualSIDPoolingV085."
            )

        for layer in range(
            num_sid_layers
        ):
            key = (
                f"pooler."
                f"sid_output_weights."
                f"{layer}"
            )
            if key not in keys:
                raise ValueError(
                    "Missing exported complete SID "
                    f"classifier slice: {key}"
                )

        if not bool(config.get("tie_word_embeddings", False)):
            for layer in range(num_sid_layers - 1):
                key = f"pooler.sid_input_embeddings.{layer}"
                if key not in keys:
                    raise ValueError(
                        "Missing exported A/B/C input embedding slice needed "
                        f"by latent/formal transitions: {key}"
                    )
            unexpected_d = "pooler.sid_input_embeddings.3"
            if unexpected_d in keys:
                raise ValueError(
                    "Unexpected D input embedding slice in layer-wise export: "
                    f"{unexpected_d}. Re-export with the new exporter."
                )

    print(
        "BRANCH-CONDITIONED INTERLEAVED LATENT MODEL CHECK PASSED"
    )


if __name__ == "__main__":
    main()
