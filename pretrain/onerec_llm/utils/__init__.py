"""Utility functions for LLM training.

This package provides general-purpose utilities including:
- Common utilities (printing, device operations, random seeds)
- Distributed training base utilities
- Data loading and processing
- Debugging and formatting tools
- Performance tracking (MFU, time tracking)
- Gradient masking
- Worker information
"""

from onerec_llm.utils.common import (
    Timer,
    dist_reduce_dict,
    get_optimizer_grouped_parameters,
    print_rank_0,
    print_rank_n,
    set_random_seed,
    to_cuda,
    to_device,
)
from onerec_llm.utils.distributed import (
    get_rank,
    get_world_size,
    get_world_size_and_rank,
    is_distributed,
)
from onerec_llm.utils.ds_utils import (
    format_dict_or_list,
    print_input_info,
    tensor_statistics,
)
from onerec_llm.utils.mfu_stats import MFUStats
from onerec_llm.utils.time_tracker import TimeTracker
from onerec_llm.utils.worker_utils import get_worker_info, pytorch_worker_info

__all__ = [
    # Common
    "Timer",
    "dist_reduce_dict",
    "get_optimizer_grouped_parameters",
    "print_rank_0",
    "print_rank_n",
    "set_random_seed",
    "to_cuda",
    "to_device",
    # Distributed
    "get_rank",
    "get_world_size",
    "get_world_size_and_rank",
    "is_distributed",
    # Debug/Format
    "format_dict_or_list",
    "print_input_info",
    "tensor_statistics",
    # Performance tracking
    "MFUStats",
    "TimeTracker",
    # Worker info
    "get_worker_info",
    "pytorch_worker_info",
]