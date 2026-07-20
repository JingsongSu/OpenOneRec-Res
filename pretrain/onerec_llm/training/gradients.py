"""Gradient computation and manipulation utilities for training.

This module provides utilities for gradient processing including:
- Gradient clipping
- Gradient norm computation for FSDP models
- Gradient masking for embedding layers in distributed training
"""

from typing import Optional

import torch
import torch.distributed as dist


def clip_grad_by_value(
    model: torch.nn.Module, 
    clip_range: Optional[float] = None
) -> None:
    """Clip gradients by value.
    
    Args:
        model: The model whose gradients will be clipped.
        clip_range: Maximum absolute value for gradients. If None, no clipping.
    """
    if clip_range is not None:
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_range)


def clip_grad_norm(
    model: torch.nn.Module,
    max_grad_norm: Optional[float] = None
) -> None:
    """Clip gradients by global L2 norm.
    
    Args:
        model: The model whose gradients will be clipped.
        max_grad_norm: Maximum allowed L2 norm. If None, no clipping.
    """
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)


def compute_fsdp_zero2_grad_norm(
    model: torch.nn.Module, 
    ignore_unused_parameters: bool = True
) -> float:
    """Compute the global L2 norm of gradients for FSDP Zero-2 models.
    
    Args:
        model: FSDP-wrapped model.
        ignore_unused_parameters: If True, ignore parameters without gradients.
    
    Returns:
        The global L2 norm of all gradients.
    """
    total_sq = torch.tensor(0.0, device=next(model.parameters()).device)
    
    for param in model.parameters():
        if param.grad is None:
            if not ignore_unused_parameters:
                raise ValueError(
                    f"Parameter {param} has no gradient. "
                    "Please check if it is being used correctly."
                )
            continue
        
        local_grad = param.grad.to_local()
        total_sq += torch.sum(local_grad ** 2)
    
    dist.all_reduce(total_sq, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
    grad_norm = torch.sqrt(total_sq).item()
    
    return grad_norm


class EmbeddingGradientMasker:
    """Freeze a portion of embedding parameters during distributed training.
    
    In distributed training with DTensor, embedding layers are sharded across ranks.
    This class freezes the first `start_optimize_embedding_index` tokens in the vocabulary,
    allowing only the remaining tokens to be optimized. This is useful for progressive
    training strategies where only a subset of the vocabulary is optimized initially.
    
    Args:
        model: The model containing embedding layers
        config: Model config with vocab_size attribute
        start_optimize_embedding_index: Index from which to start optimizing embeddings.
            Tokens before this index will be frozen. If <= 0, no masking is applied.
    """
    
    def __init__(self, model, config, start_optimize_embedding_index):
        self.model = model
        self.config = config
        self.start_optimize_embedding_index = start_optimize_embedding_index
        self.embedding_params = []  # List of (name, param) tuples for embedding layers
        self.saved_weights = {}  # Dict mapping param name -> frozen weight slice (torch.Tensor)

        if start_optimize_embedding_index > 0:
            self._find_embedding_parameters()
            self._save_initial_weights()

    def _find_embedding_parameters(self):
        """Find all embedding-related parameters (embed_tokens and lm_head)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and ("embed_tokens" in name or "lm_head" in name):
                self.embedding_params.append((name, param))

    def _save_initial_weights(self):
        """Save frozen weight slices for each rank in distributed training.
        
        In distributed training, embedding parameters are sharded across ranks.
        This method calculates which portion of the local shard needs to be frozen
        and saves those weights for later restoration after optimizer steps.
        """
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        full_vocab_size = self.config.vocab_size
        
        # Calculate shard boundaries: each rank owns a contiguous slice of the vocabulary
        shard_size = (full_vocab_size + dp_world_size - 1) // dp_world_size
        shard_offset = dp_rank * shard_size

        with torch.no_grad():
            for name, param in self.embedding_params:
                # Get local tensor from DTensor (param is a DTensor in distributed mode)
                local_param_tensor = param.to_local()
                local_shard_size = local_param_tensor.shape[0]
                
                # Calculate overlap between frozen range [0, start_optimize_embedding_index)
                # and this rank's shard [shard_offset, shard_offset + local_shard_size)
                overlap_start = shard_offset
                overlap_end = min(self.start_optimize_embedding_index, shard_offset + local_shard_size)
                
                # Number of rows in this rank's shard that need to be frozen
                num_local_rows = 0
                if overlap_end > overlap_start:
                    num_local_rows = int(overlap_end - overlap_start)

                # Save the frozen slice for restoration after optimizer steps
                if num_local_rows > 0:
                    self.saved_weights[name] = local_param_tensor[:num_local_rows].clone()

    def save_frozen_params(self):
        """Deprecated: Logic moved to __init__. Kept for backward compatibility."""
        pass

    def apply_gradient_mask(self, optimizer=None):
        """Deprecated: We use restore strategy instead. Kept for backward compatibility."""
        pass

    def restore_frozen_params(self):
        """Restore frozen parameters after optimizer.step().
        
        This should be called after each optimizer.step() to restore the frozen
        portion of embedding weights that were modified by the optimizer.
        Uses .to_local() to safely modify DTensor parameters in distributed training.
        """
        if self.start_optimize_embedding_index <= 0 or not self.saved_weights:
            return

        with torch.no_grad():
            for name, param in self.embedding_params:
                if name in self.saved_weights:
                    # Get local tensor from DTensor for modification
                    local_param_tensor = param.to_local()
                    
                    saved_slice = self.saved_weights[name]
                    num_to_restore = saved_slice.shape[0]

                    if num_to_restore > 0:
                        # Restore frozen weights by copying saved slice back
                        local_param_tensor[:num_to_restore].copy_(saved_slice)

