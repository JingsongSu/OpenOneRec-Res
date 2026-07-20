"""
MFU (Model FLOPs Utilization) Evaluator

Computes MFU metric based on:
- Model parameters
- Token statistics
- GPU hardware information
- Generation time

MFU = (num_params × 2 × total_tokens) / (gpu_flops × gpu_count × time_s)
"""

from typing import Dict, Any, Optional

from benchmark.console import console, warning_style, dim_style


def compute_mfu(
    num_params: float,
    total_tokens: int,
    gpu_tflops: float,
    gpu_count: int,
    time_seconds: float,
) -> float:
    """
    Compute Model FLOPs Utilization (MFU)

    Formula:
        MFU = (num_params × 2 × total_tokens) / (gpu_flops × gpu_count × time_s)

    Args:
        num_params: Number of model parameters
        total_tokens: Total tokens processed (input + output)
        gpu_tflops: GPU theoretical peak TFLOPS (for BF16/FP16)
        gpu_count: Number of GPUs used
        time_seconds: Total time in seconds

    Returns:
        MFU value (0-1, typically 0.01-0.5 for inference)
    """
    if time_seconds <= 0:
        console.print("⚠ Time is zero or negative, cannot compute MFU", style=warning_style)
        return 0.0

    if gpu_tflops is None or gpu_tflops <= 0:
        console.print("⚠ GPU TFLOPS is not available, cannot compute MFU", style=warning_style)
        return 0.0

    if num_params is None or num_params <= 0:
        console.print("⚠ Model parameters not specified, cannot compute MFU", style=warning_style)
        return 0.0

    # Convert TFLOPS to FLOPS
    gpu_flops = gpu_tflops * 1e12

    # Compute total FLOPs required
    # For inference: FLOPs ≈ 2 × num_params × num_tokens
    total_flops = num_params * 2 * total_tokens

    # Compute theoretical peak FLOPs available
    theoretical_flops = gpu_flops * gpu_count * time_seconds

    # MFU = actual FLOPs / theoretical FLOPs
    mfu = total_flops / theoretical_flops

    return mfu


def compute_mfu_from_generation_data(gen_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Compute MFU metrics from generation result data

    Args:
        gen_data: Generation result data from JSON file, containing:
            - num_params: Model parameters
            - mfu_stats_aggregate: MFU statistics aggregate (dict with lists)
            - hardware_info: GPU hardware info
            - total_time: Total generation time

    Returns:
        Dictionary containing MFU metrics, or None if cannot compute
        For multi-stage generation, returns MFU_stages as a list
    """
    # Extract required fields
    num_params = gen_data.get("num_params")
    mfu_stats_aggregate = gen_data.get("mfu_stats_aggregate", {})
    hardware_info = gen_data.get("hardware_info", {})
    total_time = gen_data.get("total_time", 0)

    # Validate required data
    if not num_params:
        console.print("[DEBUG] MFU: Model parameters not available in generation data, skipping MFU calculation", style=dim_style)
        return None

    if not mfu_stats_aggregate or len(mfu_stats_aggregate.get("total_time", [])) == 0:
        console.print("[DEBUG] MFU: MFU statistics not available in generation data, skipping MFU calculation", style=dim_style)
        return None

    if not hardware_info:
        console.print("[DEBUG] MFU: Hardware info not available in generation data, skipping MFU calculation", style=dim_style)
        return None

    # Extract hardware info
    gpu_tflops = hardware_info.get("gpu_tflops")
    gpu_count = hardware_info.get("gpu_count", 1)
    gpu_model = hardware_info.get("gpu_model", "unknown")

    if gpu_tflops is None:
        console.print(f"⚠ GPU TFLOPS not available for {gpu_model}, cannot compute MFU", style=warning_style)
        return None

    # Extract lists from aggregate stats
    total_input_tokens_list = mfu_stats_aggregate.get("total_input_tokens", [])
    total_output_tokens_list = mfu_stats_aggregate.get("total_output_tokens", [])
    total_time_list = mfu_stats_aggregate.get("total_time", [])

    # Validate list lengths are consistent
    if not (len(total_input_tokens_list) == len(total_output_tokens_list) == len(total_time_list)):
        console.print(
            f"⚠ Inconsistent list lengths in mfu_stats_aggregate: "
            f"input_tokens={len(total_input_tokens_list)}, "
            f"output_tokens={len(total_output_tokens_list)}, "
            f"times={len(total_time_list)}",
            style=warning_style
        )
        return None

    num_stages = len(total_time_list)

    # Compute MFU for each stage
    mfu_list = []

    for stage_idx in range(num_stages):
        stage_num = stage_idx + 1
        total_input_tokens = total_input_tokens_list[stage_idx] if stage_idx < len(total_input_tokens_list) else 0
        total_output_tokens = total_output_tokens_list[stage_idx] if stage_idx < len(total_output_tokens_list) else 0
        stage_time = total_time_list[stage_idx]
        total_tokens = total_input_tokens + total_output_tokens

        if total_tokens == 0:
            console.print(f"⚠ Stage {stage_num}: Total tokens is zero, skipping", style=warning_style)
            return None

        if stage_time <= 0:
            console.print(f"⚠ Stage {stage_num}: Stage time is zero or negative, skipping", style=warning_style)
            return None

        # Compute MFU for this stage using per-stage time
        mfu = compute_mfu(
            num_params=num_params,
            total_tokens=total_tokens,
            gpu_tflops=gpu_tflops,
            gpu_count=gpu_count,
            time_seconds=stage_time,
        )

        mfu_list.append(round(mfu, 6))

    if len(mfu_list) == 0:
        console.print("⚠ No valid stages for MFU calculation", style=warning_style)
        return None

    # Create metrics with symmetric list structure
    mfu_metrics = {
        "mfu": mfu_list,
        "gpu_model": gpu_model,
        "gpu_count": gpu_count,
        "num_params": num_params,
        "total_input_tokens": total_input_tokens_list,
        "total_output_tokens": total_output_tokens_list,
        "stage_time": total_time_list,
    }

    return mfu_metrics


