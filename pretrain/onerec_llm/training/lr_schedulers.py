"""Learning rate schedulers for training."""

import math
from functools import partial
from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
    num_stop_steps: int = 0,
    min_lr_rate: float = 0.0
) -> float:
    """Compute learning rate multiplier for cosine schedule with warmup.
    
    Args:
        current_step: Current training step.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        num_cycles: Number of cosine cycles.
        num_stop_steps: Number of steps to keep LR at 0 at the start.
        min_lr_rate: Minimum learning rate as a fraction of max LR.
    
    Returns:
        Learning rate multiplier (0.0 to 1.0).
    """
    if num_stop_steps > 0 and current_step < num_stop_steps:
        return 0.0
    
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    
    if current_step > num_training_steps:
        return min_lr_rate
    
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    factor = factor * (1 - min_lr_rate) + min_lr_rate
    return max(0.0, factor)

def get_cosine_scheduler(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    num_stop_steps: int = 0,
    last_epoch: int = -1,
    min_lr: Optional[float] = None,
    min_lr_rate: Optional[float] = None,
    **kwargs
) -> LambdaLR:
    """Create a cosine learning rate scheduler with warmup.
    
    Args:
        optimizer: Optimizer to schedule.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        num_cycles: Number of cosine cycles. Default: 0.5.
        num_stop_steps: Number of steps to keep LR at 0 at the start. Default: 0.
        last_epoch: Last epoch index for resuming. Default: -1.
        min_lr: Minimum learning rate (absolute value).
        min_lr_rate: Minimum learning rate as fraction of max LR.
    
    Returns:
        LambdaLR scheduler with cosine schedule.
    """
    if min_lr is not None and min_lr_rate is not None:
        raise ValueError("Only one of min_lr or min_lr_rate should be set")
    elif min_lr is not None:
        min_lr_rate = min_lr / optimizer.defaults["lr"]
    elif min_lr_rate is None:
        raise ValueError("One of min_lr or min_lr_rate must be set")

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_rate=min_lr_rate,
        num_stop_steps=num_stop_steps,
    )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_scheduler(
    name: str,
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    **kwargs
) -> LambdaLR:
    """Get a learning rate scheduler by name.
    
    Args:
        name: Scheduler name. Currently only supports "cosine".
        optimizer: Optimizer to schedule.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        **kwargs: Additional arguments passed to the scheduler.
    
    Returns:
        Learning rate scheduler instance.
    
    Raises:
        NotImplementedError: If scheduler name is not supported.
    """
    if name == "cosine":
        return get_cosine_scheduler(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **kwargs
        )
    else:
        raise NotImplementedError(f"Unsupported LR scheduler `{name}`")

