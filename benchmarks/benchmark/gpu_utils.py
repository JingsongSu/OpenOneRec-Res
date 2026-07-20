"""
GPU hardware detection and FLOPS calculation utilities for MFU computation.
"""

from typing import Dict, Any, Optional
from benchmark.console import console

# GPU theoretical peak FLOPS (TFLOPS) for BF16/FP16
# Source: Official vendor specifications
GPU_TFLOPS_MAP = {
    # NVIDIA A100 series
    "A100-SXM4-40GB": 312.0,
    "A100-SXM4-80GB": 312.0,
    "A100-PCIE-40GB": 312.0,
    "A100-PCIE-80GB": 312.0,

    # NVIDIA A800 series (China-specific A100 variant)
    "A800-SXM4-80GB": 312.0,
    "A800": 312.0,

    # NVIDIA H100 series
    "H100-SXM5-80GB": 989.0,
    "H100-PCIE-80GB": 756.0,
    "H100": 989.0,

    # NVIDIA V100 series
    "V100-SXM2-16GB": 125.0,
    "V100-SXM2-32GB": 125.0,
    "V100-PCIE-16GB": 112.0,
    "V100-PCIE-32GB": 112.0,

    # NVIDIA A40
    "A40": 149.7,

    # NVIDIA A30
    "A30": 165.0,

    # NVIDIA A10
    "A10": 125.0,

    # NVIDIA RTX series
    "RTX 4090": 82.6,
    "RTX 4080": 48.7,
    "RTX 3090": 35.6,
    "RTX 3080": 29.8,
}


def _normalize_gpu_name(gpu_name: str) -> str:
    """
    Normalize GPU name for lookup in TFLOPS map.

    Args:
        gpu_name: Raw GPU name from torch.cuda

    Returns:
        Normalized GPU name
    """
    gpu_name = gpu_name.strip()

    # Try exact match first
    if gpu_name in GPU_TFLOPS_MAP:
        return gpu_name

    # Try fuzzy matching
    gpu_name_upper = gpu_name.upper()

    # Match A100 variants
    if "A100" in gpu_name_upper:
        if "80GB" in gpu_name_upper or "80G" in gpu_name_upper:
            return "A100-SXM4-80GB"
        else:
            return "A100-SXM4-40GB"

    # Match A800
    if "A800" in gpu_name_upper:
        return "A800"

    # Match H100 variants
    if "H100" in gpu_name_upper:
        if "PCIE" in gpu_name_upper or "PCIe" in gpu_name_upper:
            return "H100-PCIE-80GB"
        else:
            return "H100-SXM5-80GB"

    # Match V100 variants
    if "V100" in gpu_name_upper:
        if "32GB" in gpu_name_upper or "32G" in gpu_name_upper:
            return "V100-SXM2-32GB"
        else:
            return "V100-SXM2-16GB"

    # Match other GPUs
    for known_gpu in GPU_TFLOPS_MAP.keys():
        if known_gpu.upper() in gpu_name_upper:
            return known_gpu

    return gpu_name


def get_gpu_tflops(gpu_name: str) -> Optional[float]:
    """
    Get theoretical peak TFLOPS for a given GPU model.

    Args:
        gpu_name: GPU model name

    Returns:
        TFLOPS value for BF16/FP16, or None if unknown
    """
    normalized_name = _normalize_gpu_name(gpu_name)
    return GPU_TFLOPS_MAP.get(normalized_name)


def get_gpu_info() -> Dict[str, Any]:
    """
    Detect GPU hardware information using PyTorch.

    Returns:
        Dictionary containing:
        - gpu_available: bool, whether GPU is available
        - gpu_count: int, number of GPUs
        - gpu_model: str, GPU model name
        - gpu_memory_total_gb: float, total GPU memory in GB
        - gpu_tflops: float, theoretical peak TFLOPS for BF16/FP16
    """
    try:
        import torch
    except ImportError:
        console.print("PyTorch not available, cannot detect GPU info")
        return {
            "gpu_available": False,
            "gpu_count": 0,
            "gpu_model": "unknown",
            "gpu_memory_total_gb": 0.0,
            "gpu_tflops": None,
        }

    if not torch.cuda.is_available():
        console.print("CUDA not available")
        return {
            "gpu_available": False,
            "gpu_count": 0,
            "gpu_model": "unknown",
            "gpu_memory_total_gb": 0.0,
            "gpu_tflops": None,
        }

    gpu_count = torch.cuda.device_count()

    # Get properties of the first GPU (assume homogeneous cluster)
    gpu_props = torch.cuda.get_device_properties(0)
    gpu_model = gpu_props.name
    gpu_memory_total_gb = gpu_props.total_memory / (1024 ** 3)  # Convert bytes to GB

    # Get TFLOPS
    gpu_tflops = get_gpu_tflops(gpu_model)

    if gpu_tflops is None:
        console.print(
            f"Unknown GPU model '{gpu_model}', cannot determine TFLOPS. "
            f"Please add it to GPU_TFLOPS_MAP in gpu_utils.py"
        )

    gpu_info = {
        "gpu_available": True,
        "gpu_count": gpu_count,
        "gpu_model": gpu_model,
        "gpu_memory_total_gb": round(gpu_memory_total_gb, 2),
        "gpu_tflops": gpu_tflops,
    }

    console.print(f"Detected GPU: {gpu_model} x {gpu_count}, {gpu_tflops} TFLOPS (BF16/FP16)")

    return gpu_info
