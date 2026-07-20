"""Training utilities for FSDP-based LLM training.

This package provides core training functionality including:
- Distributed training with FSDP
- Checkpoint management
- Learning rate scheduling
- Gradient computation and masking
- Activation checkpointing
"""

from onerec_llm.training.activations import set_activation_checkpointing
from onerec_llm.training.checkpoint import (
    AppState,
    DistributedCheckpointer,
    load_checkpoint_to_state_dict,
    load_hf_checkpoint,
    load_safetensors,
    safe_torch_load,
)
from onerec_llm.training.common import set_default_dtype
from onerec_llm.training.distributed import (
    load_from_full_model_state_dict,
    shard_model,
)
from onerec_llm.training.gradients import (
    EmbeddingGradientMasker,
    clip_grad_by_value,
    compute_fsdp_zero2_grad_norm,
)
from onerec_llm.training.lr_schedulers import get_cosine_scheduler, get_scheduler

__all__ = [
    # Activations
    "set_activation_checkpointing",
    # Checkpoint
    "AppState",
    "DistributedCheckpointer",
    "load_checkpoint_to_state_dict",
    "load_hf_checkpoint",
    "load_safetensors",
    "safe_torch_load",
    # Common
    "set_default_dtype",
    # Distributed
    "load_from_full_model_state_dict",
    "shard_model",
    # Gradients
    "EmbeddingGradientMasker",
    "clip_grad_by_value",
    "clip_grad_norm",
    "compute_fsdp_zero2_grad_norm",
    # LR Schedulers
    "get_cosine_scheduler",
    "get_scheduler",
]

