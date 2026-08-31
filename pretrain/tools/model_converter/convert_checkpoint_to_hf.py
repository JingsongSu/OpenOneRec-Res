"""Checkpoint to HuggingFace Format Converter.

This module converts PyTorch/DCP checkpoints to HuggingFace format and copies
the tokenizer/config sidecar files from a source HuggingFace model.

Important:
    Modern Transformers stores the default chat template in
    ``chat_template.jinja`` and optional named templates under
    ``additional_chat_templates/*.jinja``.  These files must be preserved
    during checkpoint -> HF conversion, otherwise SFT/inference code calling
    ``tokenizer.apply_chat_template`` will fail even though Stage1/Stage2
    pretraining can still run.
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import tqdm
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import (
    _EmptyStateDictLoadPlanner,
)
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


SHARD_FNAME_TEMPLATE = "model-{cpt_idx}-of-{num_shards}"
BYTES_PER_GB = 1024 * 1024 * 1024
DEFAULT_MAX_GB_PER_SHARD = 5
DEFAULT_DTYPE = "bf16"

# Explicit HuggingFace/tokenizer sidecar files.
#
# chat_template.jinja is the critical addition compared with the old version.
# tokenizer_config.json alone is not sufficient with modern Transformers,
# because save_pretrained() normally writes the chat template as a standalone
# chat_template.jinja file.
HF_CONFIG_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.jinja",
    # Legacy processor-side format. Harmless to preserve if present.
    "chat_template.json",
]

# Modern Transformers may store extra named templates here.
HF_CONFIG_DIRS = [
    "additional_chat_templates",
]

# Catch useful tokenizer/config sidecars not explicitly listed above.
# *.jinja is included so future root-level template files are preserved too.
HF_EXTRA_PATTERNS = [
    "*.json",
    "*.txt",
    "*.jinja",
]


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype name to torch dtype."""
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(
            f"Unsupported dtype: {dtype_str}. "
            f"Supported: {list(dtype_map.keys())}"
        )
    return dtype_map[dtype_str]


def _extract_state_dict_from_checkpoint(
    checkpoint: Dict,
    model_only: bool = True,
) -> Dict[str, torch.Tensor]:
    """Extract a model state dict from several common checkpoint layouts."""
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Unsupported checkpoint format: {type(checkpoint)}"
        )

    if (
        model_only
        and "app" in checkpoint
        and isinstance(checkpoint["app"], dict)
        and "model" in checkpoint["app"]
    ):
        logger.info("Found nested structure: checkpoint['app']['model']")
        return checkpoint["app"]["model"]

    if "model" in checkpoint:
        logger.info("Found structure: checkpoint['model']")
        return checkpoint["model"]

    if "state_dict" in checkpoint:
        logger.info("Found structure: checkpoint['state_dict']")
        return checkpoint["state_dict"]

    logger.info("Using entire checkpoint as state_dict")
    return checkpoint


def _convert_state_dict_to_shards(
    state_dict: Dict[str, torch.Tensor],
    output_dir: Union[str, os.PathLike],
    use_safetensor: bool = True,
    max_gb_per_shard: int = DEFAULT_MAX_GB_PER_SHARD,
    dtype: str = DEFAULT_DTYPE,
) -> None:
    """Convert a state dict to sharded HF weights."""
    torch_dtype = _get_torch_dtype(dtype)
    logger.info("Converting state_dict to %s format", dtype)

    logger.info("Converting tensor data types...")
    for key in tqdm.tqdm(
        list(state_dict.keys()),
        desc="Converting dtypes",
    ):
        tensor = state_dict[key]
        if not torch.is_tensor(tensor):
            raise TypeError(
                f"State-dict value is not a tensor: "
                f"key={key!r}, type={type(tensor)}"
            )
        state_dict[key] = tensor.to(torch_dtype)

    max_bytes_per_shard = max_gb_per_shard * BYTES_PER_GB

    split_state_dicts: Dict[
        int,
        Dict[str, torch.Tensor],
    ] = {}
    shard_idx = 0
    total_size = 0
    current_size = 0

    logger.info(
        "Splitting state_dict into shards (max %s GB per shard)...",
        max_gb_per_shard,
    )

    for key, weight in tqdm.tqdm(
        state_dict.items(),
        desc="Creating shards",
    ):
        if shard_idx not in split_state_dicts:
            split_state_dicts[shard_idx] = {}

        split_state_dicts[shard_idx][key] = weight

        weight_size = (
            weight.numel()
            * weight.element_size()
        )
        current_size += weight_size
        total_size += weight_size

        if current_size >= max_bytes_per_shard:
            shard_idx += 1
            current_size = 0

    num_shards = len(split_state_dicts)
    weight_map: Dict[str, str] = {}

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "Writing %s shard files...",
        num_shards,
    )

    for shard_idx, shard_state_dict in tqdm.tqdm(
        split_state_dicts.items(),
        desc="Writing shards",
    ):
        shard_name = SHARD_FNAME_TEMPLATE.format(
            cpt_idx=f"{shard_idx}".zfill(5),
            num_shards=f"{num_shards}".zfill(5),
        )

        if use_safetensor:
            shard_path = (
                output_path
                / f"{shard_name}.safetensors"
            )
            save_file(
                shard_state_dict,
                shard_path,
                metadata={"format": "pt"},
            )
        else:
            shard_path = (
                output_path
                / f"{shard_name}.bin"
            )
            torch.save(
                shard_state_dict,
                shard_path,
            )

        for key in shard_state_dict.keys():
            weight_map[key] = shard_path.name

        shard_size_gb = (
            os.path.getsize(shard_path)
            / BYTES_PER_GB
        )
        logger.info(
            "Shard %s/%s: %.2f GiB saved to %s",
            shard_idx + 1,
            num_shards,
            shard_size_gb,
            shard_path,
        )

    index_filename = (
        "model.safetensors.index.json"
        if use_safetensor
        else "model.bin.index.json"
    )
    index_path = (
        output_path
        / index_filename
    )

    index_data = {
        "metadata": {
            "total_size": total_size,
        },
        "weight_map": weight_map,
    }

    with open(
        index_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            index_data,
            handle,
            indent=2,
        )

    logger.info(
        "Index file saved to %s",
        index_path,
    )
    logger.info(
        "Total model size: %.2f GiB",
        total_size / BYTES_PER_GB,
    )


def pth_to_hf_format(
    pth_file_path: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    model_only: bool = True,
    use_safetensor: bool = True,
    max_gb_per_shard: int = DEFAULT_MAX_GB_PER_SHARD,
    dtype: str = DEFAULT_DTYPE,
) -> None:
    """Convert a .pth checkpoint to HuggingFace format."""
    pth_path = Path(pth_file_path)

    if not pth_path.exists():
        raise FileNotFoundError(
            f"PTH file not found: {pth_path}"
        )

    if pth_path.suffix != ".pth":
        raise ValueError(
            f"Expected .pth file, got: "
            f"{pth_path.suffix}"
        )

    logger.info(
        "Loading PTH file from %s...",
        pth_path,
    )
    checkpoint = torch.load(
        pth_path,
        map_location="cpu",
    )

    state_dict = (
        _extract_state_dict_from_checkpoint(
            checkpoint,
            model_only=model_only,
        )
    )
    logger.info(
        "Loaded state_dict with %s keys",
        len(state_dict),
    )

    _convert_state_dict_to_shards(
        state_dict=state_dict,
        output_dir=output_dir,
        use_safetensor=use_safetensor,
        max_gb_per_shard=max_gb_per_shard,
        dtype=dtype,
    )


def dcp_to_hf_format(
    dcp_checkpoint_dir: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    model_only: bool = True,
    use_safetensor: bool = True,
    max_gb_per_shard: int = DEFAULT_MAX_GB_PER_SHARD,
    dtype: str = DEFAULT_DTYPE,
) -> None:
    """Convert a DCP checkpoint directory to HuggingFace format."""
    dcp_path = Path(
        dcp_checkpoint_dir
    )

    if not dcp_path.exists():
        raise FileNotFoundError(
            f"DCP checkpoint directory not found: "
            f"{dcp_path}"
        )

    if not dcp_path.is_dir():
        raise ValueError(
            f"Expected directory, got: "
            f"{dcp_path}"
        )

    logger.info(
        "Loading DCP checkpoint from %s...",
        dcp_path,
    )

    state_dict: STATE_DICT_TYPE = {}

    _load_state_dict(
        state_dict,
        storage_reader=FileSystemReader(
            str(dcp_path)
        ),
        planner=(
            _EmptyStateDictLoadPlanner()
        ),
        no_dist=True,
    )

    logger.info(
        "DCP checkpoint loaded successfully"
    )

    if model_only:
        if (
            "app" not in state_dict
            or "model" not in state_dict["app"]
        ):
            raise ValueError(
                "Expected 'app.model' in DCP checkpoint "
                "when model_only=True"
            )

        state_dict = (
            state_dict["app"]["model"]
        )

        logger.info(
            "Extracted model state_dict with %s keys",
            len(state_dict),
        )

    _convert_state_dict_to_shards(
        state_dict=state_dict,
        output_dir=output_dir,
        use_safetensor=use_safetensor,
        max_gb_per_shard=max_gb_per_shard,
        dtype=dtype,
    )


def _read_embedded_chat_template(
    model_dir: Path,
):
    """Read a legacy chat_template field from tokenizer_config.json."""
    tokenizer_config_path = (
        model_dir
        / "tokenizer_config.json"
    )

    if not tokenizer_config_path.is_file():
        return None

    try:
        config = json.loads(
            tokenizer_config_path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return None

    return config.get(
        "chat_template"
    )


def _default_chat_template_present(
    model_dir: Path,
) -> bool:
    """Return True if the tokenizer has a usable default chat template."""
    root_template = (
        model_dir
        / "chat_template.jinja"
    )

    if (
        root_template.is_file()
        and root_template.stat().st_size > 0
    ):
        return True

    embedded = (
        _read_embedded_chat_template(
            model_dir
        )
    )

    if isinstance(
        embedded,
        str,
    ):
        return bool(
            embedded.strip()
        )

    if isinstance(
        embedded,
        list,
    ):
        for item in embedded:
            if (
                isinstance(item, dict)
                and item.get("name") == "default"
                and isinstance(
                    item.get("template"),
                    str,
                )
                and item["template"].strip()
            ):
                return True

    if isinstance(
        embedded,
        dict,
    ):
        default = embedded.get(
            "default"
        )
        return (
            isinstance(default, str)
            and bool(default.strip())
        )

    return False


def _copy_file(
    source_file: Path,
    output_path: Path,
    copied_files: set[str],
) -> None:
    """Copy one sidecar file."""
    dest_file = (
        output_path
        / source_file.name
    )

    shutil.copy2(
        source_file,
        dest_file,
    )

    copied_files.add(
        source_file.name
    )

    logger.debug(
        "Copied %s to %s",
        source_file,
        dest_file,
    )


def copy_hf_config_files(
    source_hf_model_path: Union[
        str,
        os.PathLike,
    ],
    output_dir: Union[
        str,
        os.PathLike,
    ],
) -> None:
    """Copy HuggingFace config/tokenizer/chat-template sidecar files.

    The old converter copied only known files plus ``*.json`` and ``*.txt``.
    That omitted modern ``chat_template.jinja`` files.  This implementation
    explicitly preserves the default template and optional named template
    directory, and checks that an existing source template is not lost.
    """
    source_path = Path(
        source_hf_model_path
    )
    output_path = Path(
        output_dir
    )

    if not source_path.exists():
        logger.warning(
            "Source HuggingFace model path does not exist: %s",
            source_path,
        )
        return

    if not source_path.is_dir():
        logger.warning(
            "Source path is not a directory: %s",
            source_path,
        )
        return

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_had_default_template = (
        _default_chat_template_present(
            source_path
        )
    )

    copied_files: set[str] = set()

    # Copy explicit files first.
    for config_file in HF_CONFIG_FILES:
        source_file = (
            source_path
            / config_file
        )

        if source_file.is_file():
            _copy_file(
                source_file=source_file,
                output_path=output_path,
                copied_files=copied_files,
            )

    # Preserve named templates.
    for config_dir in HF_CONFIG_DIRS:
        source_dir = (
            source_path
            / config_dir
        )

        if source_dir.is_dir():
            dest_dir = (
                output_path
                / config_dir
            )

            shutil.copytree(
                source_dir,
                dest_dir,
                dirs_exist_ok=True,
            )

            logger.debug(
                "Copied config directory: %s",
                source_dir,
            )

    # Copy additional config/tokenizer text sidecars.
    for pattern in HF_EXTRA_PATTERNS:
        for source_file in source_path.glob(
            pattern
        ):
            if not source_file.is_file():
                continue

            if source_file.name in copied_files:
                continue

            if (
                source_file.name.startswith(
                    "model-"
                )
                or source_file.suffix
                in {
                    ".bin",
                    ".safetensors",
                }
            ):
                continue

            _copy_file(
                source_file=source_file,
                output_path=output_path,
                copied_files=copied_files,
            )

    # Critical preservation check:
    # if the source tokenizer had a default template, converted must have one.
    if (
        source_had_default_template
        and not _default_chat_template_present(
            output_path
        )
    ):
        raise RuntimeError(
            "Source HuggingFace tokenizer has a chat template, "
            "but the converted output lost it. "
            f"source={source_path}, output={output_path}"
        )

    if copied_files:
        logger.info(
            "Successfully copied %s config/tokenizer files "
            "from %s to %s",
            len(copied_files),
            source_path,
            output_path,
        )
    else:
        logger.warning(
            "No config/tokenizer files found in %s",
            source_path,
        )

    if source_had_default_template:
        logger.info(
            "Chat template preserved successfully: %s",
            output_path,
        )


def get_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert PyTorch checkpoints "
            "(DCP or .pth) to HuggingFace format"
        )
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help=(
            "Path to DCP checkpoint directory "
            "or .pth file"
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=(
            "Output directory for converted "
            "HuggingFace model"
        ),
    )

    parser.add_argument(
        "--source_hf_model_path",
        type=str,
        default=None,
        help=(
            "Source HuggingFace model directory "
            "whose tokenizer/config sidecars "
            "should be copied"
        ),
    )

    parser.add_argument(
        "--use_safetensor",
        action="store_true",
        default=True,
        help=(
            "Use safetensors format "
            "(default: True)"
        ),
    )

    parser.add_argument(
        "--no_safetensor",
        dest="use_safetensor",
        action="store_false",
        help=(
            "Use .bin format instead of safetensors"
        ),
    )

    parser.add_argument(
        "--max_gb_per_shard",
        type=int,
        default=DEFAULT_MAX_GB_PER_SHARD,
        help=(
            "Maximum size per shard in GB "
            f"(default: {DEFAULT_MAX_GB_PER_SHARD})"
        ),
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default=DEFAULT_DTYPE,
        choices=[
            "fp32",
            "fp16",
            "bf16",
        ],
        help=(
            "Data type for conversion "
            f"(default: {DEFAULT_DTYPE})"
        ),
    )

    return parser


def main() -> None:
    """CLI entry point."""
    parser = get_argument_parser()
    args = parser.parse_args()

    checkpoint_path = Path(
        args.checkpoint_dir
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Checkpoint path does not exist: "
            f"{checkpoint_path}"
        )

    if (
        checkpoint_path.is_file()
        and checkpoint_path.suffix == ".pth"
    ):
        logger.info(
            "Detected PTH file: %s",
            checkpoint_path,
        )

        pth_to_hf_format(
            pth_file_path=checkpoint_path,
            output_dir=args.output_dir,
            model_only=True,
            use_safetensor=args.use_safetensor,
            max_gb_per_shard=(
                args.max_gb_per_shard
            ),
            dtype=args.dtype,
        )

    elif checkpoint_path.is_dir():
        logger.info(
            "Detected DCP checkpoint directory: %s",
            checkpoint_path,
        )

        dcp_to_hf_format(
            dcp_checkpoint_dir=checkpoint_path,
            output_dir=args.output_dir,
            model_only=True,
            use_safetensor=args.use_safetensor,
            max_gb_per_shard=(
                args.max_gb_per_shard
            ),
            dtype=args.dtype,
        )

    else:
        raise ValueError(
            f"Invalid checkpoint path: {checkpoint_path}. "
            "Expected either a .pth file or "
            "a DCP checkpoint directory."
        )

    if args.source_hf_model_path:
        logger.info(
            "Copying config/tokenizer files from %s to %s",
            args.source_hf_model_path,
            args.output_dir,
        )

        copy_hf_config_files(
            source_hf_model_path=(
                args.source_hf_model_path
            ),
            output_dir=args.output_dir,
        )

    logger.info(
        "Conversion completed successfully!"
    )


if __name__ == "__main__":
    main()
