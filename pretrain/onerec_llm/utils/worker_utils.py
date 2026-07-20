"""Worker information utilities for PyTorch DataLoader and distributed training."""

import os
import torch
import torch.distributed as dist


def get_worker_info():
    """Get PyTorch DataLoader worker information.
    
    This function prioritizes PyTorch DataLoader's worker info over environment
    variables, as it provides accurate worker information in multi-process
    DataLoader contexts.
    
    Returns:
        tuple: (worker_id, num_workers)
    """
    # Priority 1: Try to get from PyTorch DataLoader worker info
    # This is the most reliable source in DataLoader worker processes
    try:
        import torch.utils.data
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            return worker_info.id, worker_info.num_workers
    except (ModuleNotFoundError, AttributeError):
        pass
    
    # Priority 2: Fall back to environment variables (for non-DataLoader contexts)
    if "WORKER" in os.environ and "NUM_WORKERS" in os.environ:
        return int(os.environ["WORKER"]), int(os.environ["NUM_WORKERS"])
    
    # Default: single worker, worker_id = 0
    return 0, 1


def pytorch_worker_info(group=None):
    """Return node and worker info for PyTorch and some distributed environments.

    Args:
        group: Optional process group for distributed environments. Defaults to None.

    Returns:
        tuple: (rank, world_size, worker, num_workers)
    """
    # Get worker info (reuse get_worker_info to avoid code duplication)
    worker, num_workers = get_worker_info()
    
    # Get rank and world_size
    rank = 0
    world_size = 1
    
    # Check environment variables first
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        # Try to get from PyTorch distributed
        try:
            if dist.is_available() and dist.is_initialized():
                group = group or dist.group.WORLD
                rank = dist.get_rank(group=group)
                world_size = dist.get_world_size(group=group)
        except (ModuleNotFoundError, AttributeError):
            pass

    return rank, world_size, worker, num_workers

