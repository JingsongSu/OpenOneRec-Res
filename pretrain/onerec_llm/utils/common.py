"""Common utility functions for the onerec_llm package.

This module contains core utilities for:
- Distributed training (printing, reduction)
- Device operations
- Optimizer configuration
- Random seed setting
- Timing utilities
"""

import random
import time

import numpy as np
import torch
import torch.distributed as dist
from rich import print
from transformers import set_seed as set_transformers_seed

def print_rank_n(*msg, rank=0):
    try:
        _rank = dist.get_rank()
    except Exception:
        _rank = 0
    if _rank == rank:
        print(*msg)

def print_rank_0(*msg):
    print_rank_n(*msg, rank=0)

def get_optimizer_grouped_parameters(model,
                                     learning_rate: float,
                                     weight_decay,
                                     no_decay_name_list=[
                                         "bias", "LayerNorm.weight", "embedding.weight", "lm_head.weight"
                                     ],
                                     ):
    optimizer_grouped_parameters = []

    llm_wd_params_group = []
    llm_nowd_params_group = []

    for n, p in model.named_parameters():
        if p.requires_grad:
            if any(nd in n for nd in no_decay_name_list):
                # no weight decay params
                llm_nowd_params_group.append((n, p))
            else:
                llm_wd_params_group.append((n, p))
    
    # for LLM
    optimizer_grouped_parameters.append({
        "params": [p for n, p in llm_wd_params_group],
        "weight_decay": weight_decay,
        "lr": learning_rate,
    })

    optimizer_grouped_parameters.append({
        "params": [p for n, p in llm_nowd_params_group],
        "weight_decay": 0.0,
        "lr": learning_rate,
    })

    # remove empty params group
    final_optimizer_grouped_parameters = []
    for group in optimizer_grouped_parameters:
        if len(group['params']) > 0:
            final_optimizer_grouped_parameters.append(group)
    return final_optimizer_grouped_parameters

def to_device(batch, device, non_blocking=True):
    for key in list(batch.keys()):
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device=device, non_blocking=non_blocking)
    return batch

def to_cuda(batch, non_blocking=True):
    """Move batch to CUDA device. This is a convenience wrapper around to_device."""
    to_device(batch, device=torch.cuda.current_device(), non_blocking=non_blocking)

def set_random_seed(seed):
    if seed is not None:
        set_transformers_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def dist_reduce_dict(local_dict, group=None):
    gather_list = [None for _ in range(dist.get_world_size(group=group))]

    dist.all_gather_object(
        object_list=gather_list, obj=local_dict, group=group)

    def reduce_dicts(dicts):
        def _reduce(d1, d2):
            for key, value in d2.items():
                if isinstance(value, dict):
                    if key not in d1:
                        d1[key] = {}
                    _reduce(d1[key], value)
                else:
                    if key in d1:
                        d1[key] += value
                    else:
                        d1[key] = value
            return d1

        result = {}
        for d in dicts:
            result = _reduce(result, d)
        return result

    return reduce_dicts(gather_list)

class Timer:
    def __init__(self, desc: str = ""):
        self.desc = desc

    def __enter__(self):
        print_rank_0(f"Start... {self.desc}")
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()
        self.elapsed = self.end - self.start
        print_rank_0(f"End... {self.desc} elapsed: {self.elapsed:.3f} ")
