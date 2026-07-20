"""Patch converted HF config to reconstruct residual SID blocks."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model_dir", required=True)
    parser.add_argument("--residual_config", required=True)
    args = parser.parse_args()

    model_dir = Path(args.hf_model_dir)
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    residual = json.loads(
        Path(args.residual_config).read_text(encoding="utf-8")
    )
    keys = (
        "architectures",
        "residual_sid_enabled",
        "residual_sid_layer_names",
        "residual_sid_layer_starts",
        "residual_sid_layer_sizes",
        "residual_sid_begin_token_id",
        "residual_sid_end_token_id",
        "residual_sid_dropout",
    )
    for key in keys:
        config[key] = residual[key]
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (model_dir / "residual_sid_config.json").write_text(
        json.dumps(residual, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Patched {config_path}")


if __name__ == "__main__":
    main()
