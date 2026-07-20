"""Checkpoint to HuggingFace Format Converter

This module provides utilities to convert PyTorch checkpoints (DCP or .pth files)
to HuggingFace format (safetensors or bin files with sharding support).
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
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants
SHARD_FNAME_TEMPLATE = "model-{cpt_idx}-of-{num_shards}"
BYTES_PER_GB = 1024 * 1024 * 1024
DEFAULT_MAX_GB_PER_SHARD = 5
DEFAULT_DTYPE = "bf16"

# Common HuggingFace config files to copy
HF_CONFIG_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",  # SentencePiece tokenizer model file
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "generation_config.json",
    "preprocessor_config.json",  # For vision models
]


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch.dtype.
    
    Args:
        dtype_str: Data type string ("fp32", "fp16", "bf16")
        
    Returns:
        Corresponding torch.dtype
        
    Raises:
        ValueError: If dtype_str is not supported
    """
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Supported: {list(dtype_map.keys())}")
    return dtype_map[dtype_str]


def _extract_state_dict_from_checkpoint(checkpoint: Dict, model_only: bool = True) -> Dict[str, torch.Tensor]:
    """Extract state_dict from checkpoint with various structures.
    
    Args:
        checkpoint: Checkpoint dictionary
        model_only: Whether to extract only model weights
        
    Returns:
        State dictionary containing model weights
    """
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")
    
    # Check for nested DCP-like structure
    if model_only and "app" in checkpoint and "model" in checkpoint["app"]:
        logger.info("Found nested structure: checkpoint['app']['model']")
        return checkpoint["app"]["model"]
    elif "model" in checkpoint:
        logger.info("Found structure: checkpoint['model']")
        return checkpoint["model"]
    elif "state_dict" in checkpoint:
        logger.info("Found structure: checkpoint['state_dict']")
        return checkpoint["state_dict"]
    else:
        # Assume entire dict is the state_dict
        logger.info("Using entire checkpoint as state_dict")
        return checkpoint


def _convert_state_dict_to_shards(
    state_dict: Dict[str, torch.Tensor],
    output_dir: Union[str, os.PathLike],
    use_safetensor: bool = True,
    max_gb_per_shard: int = DEFAULT_MAX_GB_PER_SHARD,
    dtype: str = DEFAULT_DTYPE
) -> None:
    """Convert state_dict to sharded safetensors or bin files.
    
    Args:
        state_dict: State dictionary containing model weights
        output_dir: Output directory for sharded files
        use_safetensor: Whether to use safetensors format (default: True)
        max_gb_per_shard: Maximum size per shard in GB (default: 5)
        dtype: Data type for conversion ("fp32", "fp16", "bf16", default: "bf16")
        
    Raises:
        ValueError: If dtype is not supported
    """
    torch_dtype = _get_torch_dtype(dtype)
    logger.info(f"Converting state_dict to {dtype} format")
    
    # Convert data types
    logger.info("Converting tensor data types...")
    for key in tqdm.tqdm(state_dict.keys(), desc="Converting dtypes"):
        state_dict[key] = state_dict[key].to(torch_dtype)
    
    # Split into shards
    logger.info(f"Splitting state_dict into shards (max {max_gb_per_shard} GB per shard)...")
    split_state_dicts: Dict[int, Dict[str, torch.Tensor]] = {}
    shard_idx = 0
    total_size = 0
    current_size = 0
    
    max_bytes_per_shard = max_gb_per_shard * BYTES_PER_GB
    
    for key, weight in tqdm.tqdm(state_dict.items(), desc="Creating shards"):
        if shard_idx not in split_state_dicts:
            split_state_dicts[shard_idx] = {}
        
        split_state_dicts[shard_idx][key] = weight
        weight_size = weight.numel() * weight.element_size()
        current_size += weight_size
        total_size += weight_size
        
        if current_size >= max_bytes_per_shard:
            shard_idx += 1
            current_size = 0
    
    # Write shard files
    num_shards = len(split_state_dicts)
    weight_map: Dict[str, str] = {}
    output_path_obj = Path(output_dir)
    output_path_obj.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Writing {num_shards} shard files...")
    for shard_idx, shard_state_dict in tqdm.tqdm(split_state_dicts.items(), desc="Writing shards"):
        shard_name = SHARD_FNAME_TEMPLATE.format(
            cpt_idx=f"{shard_idx}".zfill(5),
            num_shards=f"{num_shards}".zfill(5)
        )
        
        if use_safetensor:
            shard_path = output_path_obj / f"{shard_name}.safetensors"
            save_file(shard_state_dict, shard_path, metadata={"format": "pt"})
        else:
            shard_path = output_path_obj / f"{shard_name}.bin"
            torch.save(shard_state_dict, shard_path)
        
        # Update weight map
        shard_filename = shard_path.name
        for key in shard_state_dict.keys():
            weight_map[key] = shard_filename
        
        shard_size_gb = os.path.getsize(shard_path) / BYTES_PER_GB
        logger.info(f"Shard {shard_idx + 1}/{num_shards}: {shard_size_gb:.2f} GiB saved to {shard_path}")
    
    # Write index file
    index_filename = "model.safetensors.index.json" if use_safetensor else "model.bin.index.json"
    index_path = output_path_obj / index_filename
    
    index_data = {
        "metadata": {
            "total_size": total_size
        },
        "weight_map": weight_map,
    }
    
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)
    
    logger.info(f"Index file saved to {index_path}")
    logger.info(f"Total model size: {total_size / BYTES_PER_GB:.2f} GiB")


def pth_to_hf_format(
    pth_file_path: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    model_only: bool = True,
    use_safetensor: bool = True,
    max_gb_per_shard: int = DEFAULT_MAX_GB_PER_SHARD,
    dtype: str = DEFAULT_DTYPE
) -> None:
    """Convert .pth file to HuggingFace format (safetensors or bin files).
    
    Args:
        pth_file_path: Path to .pth checkpoint file
        output_dir: Output directory for converted files
        model_only: Whether to extract only model weights (default: True)
        use_safetensor: Whether to use safetensors format (default: True)
        max_gb_per_shard: Maximum size per shard in GB (default: 5)
        dtype: Data type for conversion (default: "bf16")
        
    Raises:
        FileNotFoundError: If pth_file_path does not exist
        ValueError: If pth_file_path is not a .pth file or has unsupported format
        
    .. warning::
        To avoid OOM, it's recommended to run this function on a single rank/process.
    """
    pth_path = Path(pth_file_path)
    
    if not pth_path.exists():
        raise FileNotFoundError(f"PTH file not found: {pth_path}")
    
    if pth_path.suffix != ".pth":
        raise ValueError(f"Expected .pth file, got: {pth_path.suffix}")
    
    logger.info(f"Loading PTH file from {pth_path}...")
    checkpoint = torch.load(pth_path, map_location="cpu")
    
    # Extract state_dict from checkpoint
    state_dict = _extract_state_dict_from_checkpoint(checkpoint, model_only=model_only)
    logger.info(f"Loaded state_dict with {len(state_dict)} keys")
    
    # Convert to HuggingFace format
    _convert_state_dict_to_shards(
        state_dict=state_dict,
        output_dir=output_dir,
        use_safetensor=use_safetensor,
        max_gb_per_shard=max_gb_per_shard,
        dtype=dtype
    )


def dcp_to_hf_format(
    dcp_checkpoint_dir: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    model_only: bool = True,
    use_safetensor: bool = True,
    max_gb_per_shard: int = DEFAULT_MAX_GB_PER_SHARD,
    dtype: str = DEFAULT_DTYPE
) -> None:
    """Convert DCP (Distributed Checkpoint) to HuggingFace format.
    
    Args:
        dcp_checkpoint_dir: Directory containing the DCP checkpoint
        output_dir: Output directory for converted files
        model_only: Whether to extract only model weights (default: True)
        use_safetensor: Whether to use safetensors format (default: True)
        max_gb_per_shard: Maximum size per shard in GB (default: 5)
        dtype: Data type for conversion (default: "bf16")
        
    Raises:
        FileNotFoundError: If dcp_checkpoint_dir does not exist
        
    .. warning::
        To avoid OOM, it's recommended to run this function on a single rank/process.
    """
    dcp_path = Path(dcp_checkpoint_dir)
    
    if not dcp_path.exists():
        raise FileNotFoundError(f"DCP checkpoint directory not found: {dcp_path}")
    
    if not dcp_path.is_dir():
        raise ValueError(f"Expected directory, got: {dcp_path}")
    
    logger.info(f"Loading DCP checkpoint from {dcp_path}...")
    state_dict: STATE_DICT_TYPE = {}
    
    _load_state_dict(
        state_dict,
        storage_reader=FileSystemReader(str(dcp_path)),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    
    logger.info("DCP checkpoint loaded successfully")
    
    if model_only:
        if "app" not in state_dict or "model" not in state_dict["app"]:
            raise ValueError("Expected 'app.model' in DCP checkpoint when model_only=True")
        state_dict = state_dict["app"]["model"]
        logger.info(f"Extracted model state_dict with {len(state_dict)} keys")
    
    # Convert to HuggingFace format
    _convert_state_dict_to_shards(
        state_dict=state_dict,
        output_dir=output_dir,
        use_safetensor=use_safetensor,
        max_gb_per_shard=max_gb_per_shard,
        dtype=dtype
    )


def copy_hf_config_files(
    source_hf_model_path: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike]
) -> None:
    """Copy HuggingFace configuration files from source to output directory.
    
    Args:
        source_hf_model_path: Path to source HuggingFace model directory
        output_dir: Output directory where config files will be copied
    """
    source_path = Path(source_hf_model_path)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        logger.warning(f"Source HuggingFace model path does not exist: {source_path}")
        return
    
    if not source_path.is_dir():
        logger.warning(f"Source path is not a directory: {source_path}")
        return
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    copied_files = []
    
    # Copy known config files
    for config_file in HF_CONFIG_FILES:
        source_file = source_path / config_file
        if source_file.exists():
            dest_file = output_path / config_file
            shutil.copy2(source_file, dest_file)
            copied_files.append(config_file)
            logger.debug(f"Copied {config_file} to {output_path}")
    
    # Copy additional JSON and TXT files (may be config files)
    for pattern in ["*.json", "*.txt"]:
        for source_file in source_path.glob(pattern):
            # Skip already copied files and weight files
            if (source_file.name in copied_files or 
                source_file.name.startswith("model-") or
                source_file.suffix in [".bin", ".safetensors"]):
                continue
            
            dest_file = output_path / source_file.name
            if not dest_file.exists():  # Avoid overwriting already copied files
                shutil.copy2(source_file, dest_file)
                if source_file.name not in HF_CONFIG_FILES:
                    logger.debug(f"Copied additional file: {source_file.name}")
    
    if copied_files:
        logger.info(f"Successfully copied {len(copied_files)} config files from {source_path} to {output_path}")
    else:
        logger.warning(f"No config files found in {source_path}")


def get_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Convert PyTorch checkpoints (DCP or .pth) to HuggingFace format"
    )
    
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to DCP checkpoint directory or .pth file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for converted HuggingFace model"
    )
    
    parser.add_argument(
        "--source_hf_model_path",
        type=str,
        default=None,
        help="Path to original HuggingFace model to copy config files from (optional)"
    )
    
    parser.add_argument(
        "--use_safetensor",
        action="store_true",
        default=True,
        help="Use safetensors format (default: True)"
    )
    
    parser.add_argument(
        "--no_safetensor",
        dest="use_safetensor",
        action="store_false",
        help="Use .bin format instead of safetensors"
    )
    
    parser.add_argument(
        "--max_gb_per_shard",
        type=int,
        default=DEFAULT_MAX_GB_PER_SHARD,
        help=f"Maximum size per shard in GB (default: {DEFAULT_MAX_GB_PER_SHARD})"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default=DEFAULT_DTYPE,
        choices=["fp32", "fp16", "bf16"],
        help=f"Data type for conversion (default: {DEFAULT_DTYPE})"
    )
    
    return parser


def main() -> None:
    """Main entry point for the script."""
    parser = get_argument_parser()
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint_dir)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    # Auto-detect input type: .pth file or DCP checkpoint directory
    if checkpoint_path.is_file() and checkpoint_path.suffix == ".pth":
        logger.info(f"Detected PTH file: {checkpoint_path}")
        pth_to_hf_format(
            pth_file_path=checkpoint_path,
            output_dir=args.output_dir,
            model_only=True,
            use_safetensor=args.use_safetensor,
            max_gb_per_shard=args.max_gb_per_shard,
            dtype=args.dtype
        )
    elif checkpoint_path.is_dir():
        logger.info(f"Detected DCP checkpoint directory: {checkpoint_path}")
        dcp_to_hf_format(
            dcp_checkpoint_dir=checkpoint_path,
            output_dir=args.output_dir,
            model_only=True,
            use_safetensor=args.use_safetensor,
            max_gb_per_shard=args.max_gb_per_shard,
            dtype=args.dtype
        )
    else:
        raise ValueError(
            f"Invalid checkpoint path: {checkpoint_path}. "
            "Expected either a .pth file or a DCP checkpoint directory."
        )
    
    # Copy config files if source model path is provided
    if args.source_hf_model_path:
        logger.info(f"Copying config files from {args.source_hf_model_path} to {args.output_dir}")
        copy_hf_config_files(
            source_hf_model_path=args.source_hf_model_path,
            output_dir=args.output_dir
        )
    
    logger.info("Conversion completed successfully!")


if __name__ == "__main__":
    main()
