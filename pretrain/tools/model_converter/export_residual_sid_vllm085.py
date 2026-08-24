"""Export branch-conditioned interleaved residual SID weights for vLLM 0.8.5.

The pooler runs on tensor-parallel rank 0 and needs complete SID classifier
slices. Untied models need A/B/C input embedding slices because every B/C/D
interleaved thought and formal transition consumes the actual previous hard SID.
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict
import torch
from safetensors import safe_open
from safetensors.torch import save_file

CUSTOM_FILE = "model-residual-sid-vllm085.safetensors"
CUSTOM_PREFIXES = (
    "pooler.sid_output_weights.",
    "pooler.sid_input_embeddings.",
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_model_dir", required=True)
    parser.add_argument("--output_model_dir", required=True)
    parser.add_argument("--beam_size", required=True, type=int)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()

def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def copy_model_tree(source: Path, output: Path, overwrite: bool) -> None:
    if source.resolve() == output.resolve():
        return
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output} exists; pass --overwrite to replace it."
            )
        shutil.rmtree(output)
    shutil.copytree(source, output)

def discover_weight_map(model_dir: Path) -> tuple[dict[str, str], Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = read_json(index_path)
        return dict(index["weight_map"]), index_path
    weight_map: Dict[str, str] = {}
    files = sorted(
        path
        for path in model_dir.glob("*.safetensors")
        if path.name != CUSTOM_FILE
    )
    if not files:
        raise FileNotFoundError(
            f"No safetensors files were found in {model_dir}."
        )
    for file_path in files:
        with safe_open(
            file_path,
            framework="pt",
            device="cpu",
        ) as handle:
            for key in handle.keys():
                if key in weight_map:
                    raise ValueError(f"Duplicate tensor key: {key}")
                weight_map[key] = file_path.name
    return weight_map, index_path

def tensor_slice(
    model_dir: Path,
    weight_map: dict[str, str],
    key: str,
    start: int,
    end: int,
) -> torch.Tensor:
    if key not in weight_map:
        raise KeyError(f"Tensor {key!r} is absent from the checkpoint.")
    with safe_open(
        model_dir / weight_map[key],
        framework="pt",
        device="cpu",
    ) as handle:
        tensor = handle.get_slice(key)[start:end]
    return tensor.contiguous()

def update_index(
    model_dir: Path,
    weight_map: dict[str, str],
    index_path: Path,
    custom_tensors: dict[str, torch.Tensor],
) -> None:
    clean_map = {
        key: filename
        for key, filename in weight_map.items()
        if filename != CUSTOM_FILE
        and not key.startswith(CUSTOM_PREFIXES)
    }
    for key in custom_tensors:
        clean_map[key] = CUSTOM_FILE
    total_size = sum(
        (model_dir / filename).stat().st_size
        for filename in sorted(set(clean_map.values()))
        if (model_dir / filename).is_file()
    )
    index_path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": int(total_size)},
                "weight_map": dict(sorted(clean_map.items())),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

def main() -> None:
    args = parse_args()
    if args.beam_size <= 0:
        raise ValueError("--beam_size must be positive.")
    source = Path(args.source_model_dir).expanduser().resolve()
    output = Path(args.output_model_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    copy_model_tree(source, output, args.overwrite)
    config_path = output / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = read_json(config_path)
    required = (
        "residual_sid_layer_starts",
        "residual_sid_layer_sizes",
        "residual_sid_begin_token_id",
        "residual_sid_end_token_id",
        "latent_reasoning_mode",
        "latent_reasoning_num_steps",
        "latent_reasoning_num_transitions",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(
            "Run patch_residual_sid_hf_config.py first; config is missing "
            f"{missing}."
        )
    if config.get("latent_reasoning_mode") != "branch_conditioned_interleaved":
        raise ValueError(
            "This exporter expects "
            "latent_reasoning_mode='branch_conditioned_interleaved'."
        )
    starts = [int(x) for x in config["residual_sid_layer_starts"]]
    sizes = [int(x) for x in config["residual_sid_layer_sizes"]]
    if len(starts) != 4 or len(sizes) != 4:
        raise ValueError(
            "This experiment expects exactly four SID layers; "
            f"found starts={starts}, sizes={sizes}."
        )
    if int(config.get("latent_reasoning_num_steps", 0)) != 3:
        raise ValueError(
            "Branch-conditioned interleaved reasoning requires three latent steps."
        )
    if int(config.get("latent_reasoning_num_transitions", 0)) != 3:
        raise ValueError(
            "Four SID layers require exactly three interleaved latent transitions."
        )
    if args.beam_size > sizes[0]:
        raise ValueError(
            f"beam_size={args.beam_size} exceeds first-layer size={sizes[0]}."
        )
    weight_map, index_path = discover_weight_map(output)
    if not any(key.startswith("sid_residual_blocks.") for key in weight_map):
        raise ValueError(
            "No sid_residual_blocks.* weights found. Convert the integrated "
            "four-layer residual-SID SFT checkpoint."
        )
    if not any(key.startswith("latent_reasoning_blocks.") for key in weight_map):
        raise ValueError(
            "No latent_reasoning_blocks.* weights found. Convert the trained "
            "layer-wise latent-anchor checkpoint first."
        )
    input_key = "model.embed_tokens.weight"
    tied = bool(config.get("tie_word_embeddings", False))
    if "lm_head.weight" in weight_map:
        output_key = "lm_head.weight"
    elif tied and input_key in weight_map:
        output_key = input_key
    else:
        raise KeyError(
            "lm_head.weight is missing and embeddings are not tied."
        )
    custom_tensors: Dict[str, torch.Tensor] = {}
    for layer, (start, size) in enumerate(zip(starts, sizes)):
        custom_tensors[f"pooler.sid_output_weights.{layer}"] = tensor_slice(
            output,
            weight_map,
            output_key,
            start,
            start + size,
        )
    if not tied:
        # Both formal residual decoding and latent transitions only consume
        # previous-token embeddings A/B/C. D is terminal.
        for layer, (start, size) in enumerate(
            zip(starts[:-1], sizes[:-1])
        ):
            custom_tensors[
                f"pooler.sid_input_embeddings.{layer}"
            ] = tensor_slice(
                output,
                weight_map,
                input_key,
                start,
                start + size,
            )
    custom_path = output / CUSTOM_FILE
    if custom_path.exists():
        custom_path.unlink()
    save_file(custom_tensors, str(custom_path))
    update_index(output, weight_map, index_path, custom_tensors)
    config.update(
        {
            "architectures": ["Qwen3ForResidualSIDPoolingV085"],
            "residual_sid_vllm_export_version": 4,
            "residual_sid_vllm_target_version": "0.8.5",
            "residual_sid_beam_size": int(args.beam_size),
            "residual_sid_num_layers": 4,
            "residual_sid_output_stride": 5,
            "residual_sid_output_format": (
                "[beam, 5] rows: [sid0, sid1, sid2, sid3, cumulative_log_score]"
            ),
            "residual_sid_replicated_output_slices": True,
            "residual_sid_replicated_input_slices": not tied,
            "branch_interleaved_hard_sid_input_slices": (
                0 if tied else 3
            ),
        }
    )
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "source_model_dir": str(source),
        "output_model_dir": str(output),
        "target_vllm": "0.8.5",
        "beam_size": args.beam_size,
        "num_sid_layers": 4,
        "layer_starts": starts,
        "layer_sizes": sizes,
        "tie_word_embeddings": tied,
        "latent_reasoning_mode": config.get("latent_reasoning_mode"),
        "latent_reasoning_num_steps": config.get("latent_reasoning_num_steps"),
        "latent_reasoning_conditioning": config.get("latent_reasoning_conditioning"),
        "latent_reasoning_update": config.get("latent_reasoning_update"),
        "custom_file": CUSTOM_FILE,
        "custom_tensor_shapes": {
            key: list(value.shape)
            for key, value in custom_tensors.items()
        },
    }
    (output / "residual_sid_vllm085_export.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
