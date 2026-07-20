"""Model FLOPs Utilization (MFU) statistics and calculation utilities.

This module provides functionality to calculate FLOPs (Floating Point Operations)
for transformer models and compute MFU metrics for training performance monitoring.
"""

import collections
import json
import os
import platform
import re
import subprocess
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Optional, Union

import easydict


def _sum_if_list(x: Union[int, List[int]]) -> int:
    """Sum if input is a list, otherwise return as-is."""
    return sum(x) if isinstance(x, list) else x


@lru_cache(maxsize=1)
def _get_gpu_model() -> str:
    """Get NVIDIA GPU model name.
    
    Returns:
        GPU model name, or "Unknown" if detection fails.
    """
    try:
        # Try nvidia-smi (most reliable method)
        if platform.system() in ["Linux", "Darwin"]:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
        
        elif platform.system() == "Windows":
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                shell=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
            
            # Fallback: Windows Management Instrumentation
            try:
                import wmi
                c = wmi.WMI()
                gpus = c.Win32_VideoController()
                for gpu in gpus:
                    if "NVIDIA" in gpu.Name:
                        return gpu.Name
            except ImportError:
                pass
        
        # Fallback: PyTorch CUDA
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except ImportError:
            pass
        
        # Fallback: TensorFlow
        try:
            import tensorflow as tf
            if tf.test.is_gpu_available():
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    details = tf.config.experimental.get_device_details(gpus[0])
                    return details.get('device_name', 'NVIDIA GPU')
        except ImportError:
            pass
        
        # Last resort: Check Linux driver file
        if platform.system() == "Linux":
            if os.path.exists("/proc/driver/nvidia/version"):
                with open("/proc/driver/nvidia/version", "r") as f:
                    first_line = f.readline().strip()
                    match = re.search(r"NVIDIA driver \S+ for (\S+)", first_line)
                    if match:
                        return match.group(1)
    
    except Exception:
        pass
    
    return "Unknown"


@lru_cache(maxsize=1)
def _is_h800() -> bool:
    """Check if GPU is NVIDIA H800."""
    gpu_model = _get_gpu_model()
    return gpu_model.split('\n')[0].strip() == 'NVIDIA H800'


@lru_cache(maxsize=1)
def _get_gpu_flops() -> float:
    """Get theoretical peak FLOPS for current GPU.
    
    Returns:
        Peak FLOPS (H800: 989 TFLOPS, others: 312 TFLOPS)
    """
    return 989e12 if _is_h800() else 312e12


def _calculate_decoder_layer_flops(
    num_head: int,
    head_dim: int,
    hidden_size: int,
    intermediate_size: int,
    kv_heads: Optional[int] = None,
    is_causal: bool = False,
    seq_len: Union[int, List[int]] = 1,
    batch_size: int = 1,
    linear_factor: int = 2,
    attn_output_layers: int = 2
) -> Dict:
    """Calculate FLOPs for a single transformer decoder layer.
    
    Args:
        num_head: Number of attention heads
        head_dim: Dimension per attention head
        hidden_size: Hidden layer size
        intermediate_size: FFN intermediate layer size
        kv_heads: Number of KV attention heads (for Group Attention)
        is_causal: Whether to use causal masking
        seq_len: Input sequence length (int or list for variable lengths)
        batch_size: Batch size
        linear_factor: Linear computation factor (default: 2 for multiply-add)
        attn_output_layers: Number of attention output layers
    
    Returns:
        Dictionary containing FLOPs breakdown and total FLOPs
    """
    if kv_heads is None:
        kv_heads = num_head
    
    seq_len_per_sample = None if isinstance(seq_len, list) else seq_len // batch_size
    total_seq_len = _sum_if_list(seq_len)
    
    # QKV projection FLOPs
    q_flops = linear_factor * total_seq_len * hidden_size * (num_head * head_dim)
    k_flops = linear_factor * total_seq_len * hidden_size * (kv_heads * head_dim)
    v_flops = linear_factor * total_seq_len * hidden_size * (kv_heads * head_dim)
    
    # Attention scores FLOPs
    if isinstance(seq_len, list):
        attn_scores_flops = 0
        for seq_len_per_sample in seq_len:
            attn_scores_flops += (
                linear_factor * num_head * seq_len_per_sample * 
                seq_len_per_sample * head_dim
            )
    else:
        attn_scores_flops = (
            linear_factor * num_head * seq_len_per_sample * 
            seq_len_per_sample * head_dim * batch_size
        )
    
    # Causal masking reduces computation by half
    if is_causal:
        attn_scores_flops *= 0.5
    
    attn_v_flops = attn_scores_flops
    
    # Attention output projection
    attn_out_flops = linear_factor * total_seq_len * (num_head * head_dim) * hidden_size
    
    # Total attention FLOPs
    attention_flops = q_flops + k_flops + v_flops + attn_scores_flops + attn_v_flops + attn_out_flops
    
    # FFN FLOPs
    ffn_flops = (
        linear_factor * total_seq_len * hidden_size * 
        intermediate_size * attn_output_layers
    )
    
    total_flops = attention_flops + ffn_flops
    
    return {
        'total_flops': total_flops,
        'attention': {
            'q_proj': q_flops,
            'k_proj': k_flops,
            'v_proj': v_flops,
            'attn_scores': attn_scores_flops,
            'attn_v': attn_v_flops,
            'attn_out': attn_out_flops,
            'total': attention_flops
        },
        'ffn_flops': ffn_flops,
        'batch_info': {
            'batch_size': batch_size,
            'seq_len_per_sample': seq_len_per_sample
        }
    }


def _calculate_decoder_layers_flops(
    num_head: int,
    head_dim: int,
    hidden_size: int,
    intermediate_size: int,
    kv_heads: Optional[int] = None,
    is_causal: bool = False,
    seq_len: Union[int, List[int]] = 1,
    num_layers: int = 1,
    linear_factor: int = 2,
    batch_size: int = 1,
    attn_output_layers: int = 2
) -> Dict:
    """Calculate FLOPs for multiple transformer decoder layers.
    
    Args:
        num_head: Number of attention heads
        head_dim: Dimension per attention head
        hidden_size: Hidden layer size
        intermediate_size: FFN intermediate layer size
        kv_heads: Number of KV attention heads
        is_causal: Whether to use causal masking
        seq_len: Input sequence length
        num_layers: Number of decoder layers
        linear_factor: Linear computation factor
        batch_size: Batch size
        attn_output_layers: Number of attention output layers
    
    Returns:
        Dictionary containing per-layer and total FLOPs
    """
    layers_flops = []
    total_flops = 0
    
    for layer_idx in range(num_layers):
        layer_flops = _calculate_decoder_layer_flops(
            num_head=num_head,
            head_dim=head_dim,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            kv_heads=kv_heads,
            is_causal=is_causal,
            seq_len=seq_len,
            linear_factor=linear_factor,
            batch_size=batch_size,
            attn_output_layers=attn_output_layers
        )
        layers_flops.append({
            'layer_index': layer_idx,
            **layer_flops
        })
        total_flops += layer_flops['total_flops']
    
    return {
        'total_flops': total_flops,
        'per_layer_flops': layers_flops[0] if layers_flops else {},
        'avg_flops_per_layer': total_flops / num_layers if num_layers > 0 else 0,
        'num_layers': num_layers,
    }


def _calculate_llm_flops(llm_params: easydict.EasyDict) -> Dict:
    """Calculate total FLOPs for an LLM model.
    
    Args:
        llm_params: Model parameters (EasyDict with model config)
    
    Returns:
        Dictionary containing total FLOPs including LM head
    """
    linear_factor = 2
    
    llm_flops = _calculate_decoder_layers_flops(
        num_head=llm_params.num_head,
        head_dim=llm_params.head_dim,
        hidden_size=llm_params.hidden_size,
        intermediate_size=llm_params.intermediate_size,
        num_layers=llm_params.num_layers,
        kv_heads=llm_params.get('kv_heads', None),
        is_causal=llm_params.get('is_causal', True),
        seq_len=llm_params.seq_len,
        batch_size=llm_params.get('batch_size', 1),
        linear_factor=linear_factor,
        attn_output_layers=3
    )
    
    # Add LM head FLOPs
    lm_head_flops = (
        linear_factor * _sum_if_list(llm_params.seq_len) * 
        (llm_params.hidden_size * llm_params.vocab_size)
    )
    llm_flops['total_flops'] += lm_head_flops
    llm_flops['lm_head_flops'] = lm_head_flops
    
    return llm_flops


@lru_cache(maxsize=32)
def _extract_model_params(config_path: str) -> easydict.EasyDict:
    """Extract transformer parameters from model config JSON.
    
    Supports Qwen3 architecture.
    
    Args:
        config_path: Path to JSON config file
    
    Returns:
        EasyDict containing transformer parameters
    
    Raises:
        ValueError: If architecture is not supported
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if 'architectures' in config and 'Qwen3ForCausalLM' in config['architectures']:
        transformer_params = {
            'num_head': config['num_attention_heads'],
            'head_dim': config['head_dim'],
            'hidden_size': config['hidden_size'],
            'intermediate_size': config['intermediate_size'],
            'kv_heads': config['num_key_value_heads'],
            'num_layers': config['num_hidden_layers'],
            'vocab_size': config['vocab_size']
        }
    else:
        raise ValueError(
            f'Unsupported architecture. Expected Qwen3ForCausalLM, '
            f'got: {config.get("architectures", "unknown")}'
        )
    
    return easydict.EasyDict(transformer_params)


def _calc_mfu(
    config_path: str,
    total_seq_len: int,
    llm_batch_size: int = 1,
    secs_per_step: Optional[float] = None,
    _gpu_flops: Optional[float] = None
) -> Dict:
    """Calculate Model FLOPs Utilization (MFU) for LLM models.
    
    Args:
        config_path: Path to model config JSON
        total_seq_len: Total sequence length
        llm_batch_size: Batch size for LLM
        secs_per_step: Seconds per training step
        _gpu_flops: GPU peak FLOPS (auto-detected if None)
    
    Returns:
        Dictionary containing MFU metrics and FLOPs breakdown
    """
    transformer_params = _extract_model_params(config_path)
    
    # Calculate LLM FLOPs
    llm_params = easydict.EasyDict({
        **transformer_params,
        'is_causal': True,
        'seq_len': total_seq_len,
        'batch_size': llm_batch_size
    })
    
    flops = _calculate_llm_flops(llm_params)
    gpu_flops = _get_gpu_flops() if _gpu_flops is None else _gpu_flops
    
    # Add MFU metrics
    flops['total_flops*3(T)'] = flops['total_flops'] * 3 / 1e12
    flops['total_flops/gpu_flops'] = flops['total_flops'] * 3 / gpu_flops
    flops['gpu_flops'] = gpu_flops
    flops['llm_total_flops*3(T)'] = flops['total_flops*3(T)']
    flops['llm_percentage'] = 100
    
    flops['input_args'] = easydict.EasyDict(
        config_path=config_path,
        total_seq_len=total_seq_len,
        llm_batch_size=llm_batch_size,
        secs_per_step=secs_per_step
    )
    
    if secs_per_step is not None:
        flops['mfu'] = flops['total_flops/gpu_flops'] / secs_per_step
    
    return flops


class MFUStats:
    """Model FLOPs Utilization statistics tracker for LLM training.
    
    Tracks token counts and computes MFU metrics for training performance monitoring.
    
    Args:
        args: Training arguments containing model_dir and logging_per_step
    """
    
    def __init__(self, args):
        self.tokens_for_mfu = collections.defaultdict(int)
        self.mfu_per_step_per_gpu = None
        self.args = args
        self.total_mfu = defaultdict(int)
    
    def set(self, num_tokens: int, num_samples: int) -> None:
        """Accumulate token and sample counts for MFU calculation.
        
        Args:
            num_tokens: Total number of tokens
            num_samples: Number of samples
        """
        self.tokens_for_mfu["num_tokens"] += int(num_tokens)
        self.tokens_for_mfu["num_samples"] += int(num_samples)
    
    def mfu(self, secs: float, global_step: int) -> Dict[str, float]:
        """Compute MFU metrics for the current logging period.
        
        Args:
            secs: Total seconds elapsed in this period
            global_step: Current global training step
        
        Returns:
            Dictionary containing MFU metrics for logging
        """
        args = self.args
        tokens_for_mfu = self.tokens_for_mfu
        
        # Calculate MFU arguments for text-only LLM
        mfu_args = easydict.EasyDict(
            total_seq_len=round(tokens_for_mfu["num_tokens"] / args.logging_per_step),
            llm_batch_size=round(tokens_for_mfu["num_samples"] / args.logging_per_step),
            secs_per_step=secs / args.logging_per_step
        )
        
        config_path = os.path.join(args.model_dir, "config.json")
        mfu_per_step_per_gpu = _calc_mfu(config_path, **mfu_args)
        self.mfu_per_step_per_gpu = mfu_per_step_per_gpu
        
        # Accumulate total MFU
        total_mfu = self.total_mfu
        total_mfu['llm_total_flops*3(T)'] += (
            mfu_per_step_per_gpu['llm_total_flops*3(T)'] * args.logging_per_step
        )
        total_mfu['mfu'] += mfu_per_step_per_gpu['mfu'] * args.logging_per_step
        
        # Build logging dictionary
        # Current metrics: period-based MFU (current logging period)
        # Average metrics: cumulative MFU (average over entire training, smoothed)
        mfu_log_dict = {
            "perf/mfu_per_step_per_gpu_current": mfu_per_step_per_gpu['mfu'],
            "perf/llm_flops_per_step_per_gpu_current": mfu_per_step_per_gpu['llm_total_flops*3(T)'],
            "perf/mfu_per_step_per_gpu_avg": total_mfu['mfu'] / global_step,
            "perf/llm_flops_per_step_per_gpu_avg": total_mfu['llm_total_flops*3(T)'] / global_step,
        }
        
        # Reset counters for next period
        self.tokens_for_mfu = collections.defaultdict(int)
        
        return mfu_log_dict

