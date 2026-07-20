"""Distributed training base utilities.

This module provides fundamental distributed training utilities that can be used
across different modules without creating circular dependencies. For FSDP-specific
utilities, see onerec_llm.training.distributed.
"""

import os
from typing import Tuple

import torch
import torch.distributed as dist


def get_world_size_and_rank() -> Tuple[int, int]:
    """Get the current world size and rank number.
    
    This function checks multiple sources in order:
    1. PyTorch distributed (if initialized)
    2. Environment variables (RANK, WORLD_SIZE)
    3. Defaults to single process (1, 0)
    
    Returns:
        Tuple of (world_size, rank).
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(), torch.distributed.get_rank()
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])
    else:
        return 1, 0


def get_rank() -> int:
    """Get the current process rank.
    
    Returns:
        Process rank (0-based).
    """
    _, rank = get_world_size_and_rank()
    return rank


def get_world_size() -> int:
    """Get the current world size.
    
    Returns:
        Number of processes in the distributed group.
    """
    world_size, _ = get_world_size_and_rank()
    return world_size


def is_distributed() -> bool:
    """Check if distributed training is initialized.
    
    Returns:
        True if distributed training is available and initialized.
    """
    return torch.distributed.is_available() and torch.distributed.is_initialized()

