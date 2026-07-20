"""Common training utilities for distributed model training."""

from typing import Generator

import contextlib
import torch


@contextlib.contextmanager
def set_default_dtype(dtype: torch.dtype) -> Generator[None, None, None]:
    """Temporarily set torch's default dtype.
    
    Args:
        dtype: The desired default dtype.
    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)
