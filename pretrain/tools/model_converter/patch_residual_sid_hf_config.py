"""Patch converted HF config for residual SID + layer-wise latent anchors."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_model_dir",
        required=True,
    )
    parser.add_argument(
        "--residual_config",
        required=True,
    )
    args = parser.parse_args()

    model_dir = Path(
        args.hf_model_dir
    )
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
    residual = json.loads(
        Path(
            args.residual_config
        ).read_text(
            encoding="utf-8"
        )
    )

    required_keys = (
        "architectures",
        "residual_sid_enabled",
        "residual_sid_layer_names",
        "residual_sid_layer_starts",
        "residual_sid_layer_sizes",
        "residual_sid_begin_token_id",
        "residual_sid_end_token_id",
        "residual_sid_dropout",
    )

    for key in required_keys:
        if key not in residual:
            raise KeyError(
                "Missing required residual "
                f"config key: {key}"
            )
        config[key] = residual[key]

    optional_keys = (
        "residual_sid_loss_weight",
        "mask_residual_sid_lm_loss",
        "latent_reasoning_enabled",
        "latent_reasoning_mode",
        "latent_reasoning_num_steps",
        "latent_reasoning_num_transitions",
        "latent_reasoning_dropout",
        "latent_reasoning_conditioning",
        "latent_reasoning_update",
        "latent_reasoning_loss_type",
        "latent_reasoning_loss_weight",
    )

    for key in optional_keys:
        if key in residual:
            config[key] = residual[key]

    # Remove stale metadata from the previous literal-token latent experiment.
    # The tokenizer may still contain those three unused vocabulary rows when
    # reusing a latent3-expanded base model, but this experiment never injects or reads
    # them.  Keeping the config clean prevents evaluation/export code from
    # accidentally selecting the old token-latent path.
    for stale_key in (
        "latent_reasoning_tokens",
        "latent_reasoning_token_ids",
        "latent_reasoning_num_tokens",
        "mask_latent_reasoning_lm_loss",
        "latent_reasoning_scale",
        "latent_reasoning_top_k",
        "latent_reasoning_temperature",
    ):
        config.pop(stale_key, None)

    if config.get(
        "latent_reasoning_enabled",
        False,
    ):
        mode = config.get(
            "latent_reasoning_mode"
        )
        if (
            mode
            != "branch_conditioned_interleaved"
        ):
            raise ValueError(
                "Expected "
                "latent_reasoning_mode="
                "'branch_conditioned_interleaved', "
                f"got {mode!r}."
            )

        num_steps = int(
            config.get(
                "latent_reasoning_num_steps",
                0,
            )
        )
        sid_layers = len(config.get("residual_sid_layer_starts", []))
        expected_steps = sid_layers - 1
        if num_steps != expected_steps:
            raise ValueError(
                "Branch-conditioned interleaved reasoning requires one "
                "latent step before B/C/D: "
                f"steps={num_steps}, expected={expected_steps}."
            )
        transitions = int(
            config.get(
                "latent_reasoning_num_transitions",
                0,
            )
        )
        if transitions != expected_steps:
            raise ValueError(
                "Expected exactly sid_layers-1 interleaved latent blocks: "
                f"transitions={transitions}, expected={expected_steps}."
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
                "Branch-conditioned interleaved reasoning uses no auxiliary "
                "latent CE; latent_reasoning_loss_weight must be 0.0."
            )

    config_path.write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (
        model_dir
        / "residual_sid_config.json"
    ).write_text(
        json.dumps(
            residual,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Patched {config_path}"
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


if __name__ == "__main__":
    main()
