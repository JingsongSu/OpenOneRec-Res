"""Debug and formatting utilities for data structures and tensors."""

import math
import os
import traceback
from dataclasses import is_dataclass, asdict
from typing import Any, Dict, List, Tuple, Union

import torch


def convert_dataclass_to_dict(obj: Any) -> Any:
    """Convert dataclass instance to dict, return other objects unchanged."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return obj


def tensor_statistics(tensor: torch.Tensor, n: int = -1, **kwargs) -> Tuple[str, str, str, str]:
    """Compute tensor statistics at 4 granularity levels.
    
    Args:
        tensor: PyTorch tensor of any shape
        n: Partial range: -1 for first half, >0 for first n elements
    
    Returns:
        Tuple of 4 formatted stat strings: full, partial, magnitude-based, 1/10 magnitude-based
    """
    flattened = tensor.reshape(-1)
    total_elements = flattened.numel()
    
    if total_elements == 0:
        base = "mean: NaN, variance: NaN, max: NaN, min: NaN, non-zeros: 0"
        return (
            f"Full - {base}",
            f"Partial - {base}",
            f"Magnitude-based - {base}",
            f"1/10 Magnitude-based - {base}"
        )
    
    if n == -1:
        part_count = (total_elements + 1) // 2
        part_tensor = flattened[:part_count]
        part_label = f"first half ({part_count} elements)"
    elif isinstance(n, int) and n > 0:
        if n > total_elements:
            raise ValueError(f"n={n} exceeds total elements ({total_elements})")
        part_count = n
        part_tensor = flattened[:n]
        part_label = f"first {n} elements"
    else:
        raise ValueError(f"n must be -1 or positive integer, got: {n}")
    
    if total_elements <= 1:
        mag_count = 0
        mag_label = "no elements (total <= 1)"
        mag_tensor = flattened[:0]
    else:
        log_val = math.log10(total_elements)
        k = int(log_val) - 1 if log_val.is_integer() else math.floor(log_val)
        mag_count = 10 ** k
        mag_count = min(mag_count, total_elements)
        mag_tensor = flattened[:mag_count]
        mag_label = f"first {mag_count} elements (magnitude-based)"
    
    line4_count = mag_count // 10
    if line4_count <= 0:
        line4_label = "no elements (1/10 of magnitude-based <= 0)"
        line4_tensor = flattened[:0]
    else:
        line4_count = min(line4_count, total_elements)
        line4_tensor = flattened[:line4_count]
        line4_label = f"first {line4_count} elements (1/10 of magnitude-based)"
    
    def calc_stats(t: torch.Tensor) -> Tuple[float, float, float, float, int]:
        """Calculate mean, variance, max, min, non-zero count."""
        if t.numel() == 0:
            return (float('nan'), float('nan'), float('nan'), float('nan'), 0)
        return (
            torch.mean(t.float()).item(),
            torch.var(t.float(), unbiased=False).item(),
            torch.max(t).item(),
            torch.min(t).item(),
            torch.count_nonzero(t).item()
        )
    
    full_mean, full_var, full_max, full_min, full_nonzero = calc_stats(flattened)
    part_mean, part_var, part_max, part_min, part_nonzero = calc_stats(part_tensor)
    mag_mean, mag_var, mag_max, mag_min, mag_nonzero = calc_stats(mag_tensor)
    line4_mean, line4_var, line4_max, line4_min, line4_nonzero = calc_stats(line4_tensor)
    
    def format_line(label: str, mean: float, var: float, max_val: float, 
                   min_val: float, nonzero: int) -> str:
        return (f"{label} - mean: {mean:.6f}, variance: {var:.6f}, "
                f"max: {max_val:.6f}, min: {min_val:.6f}, non-zeros: {nonzero}")
    
    line1 = format_line("Full", full_mean, full_var, full_max, full_min, full_nonzero)
    line2 = format_line(part_label, part_mean, part_var, part_max, part_min, part_nonzero)
    line3 = format_line(mag_label, mag_mean, mag_var, mag_max, mag_min, mag_nonzero)
    line4 = format_line(line4_label, line4_mean, line4_var, line4_max, line4_min, line4_nonzero)
    
    return line1, line2, line3, line4


def print_input_info(
    data: Any, 
    prefix: str = "", 
    max_str_len: int = 50, 
    return_str: bool = False, 
    max_show: int = 4, 
    save_path: Union[str, None] = None, 
    **kwargs
) -> Union[None, str]:
    """Recursively print or return detailed information about input data.
    
    Supports Tensor, dict, list, tuple, str, int, float. Can save data to disk.
    
    Args:
        data: Data to print
        prefix: Prefix for each line (indentation)
        max_str_len: Max string display length
        return_str: Return string instead of printing
        max_show: Max elements for tensor preview
        save_path: Optional path to save data (tensors detached to CPU)
        **kwargs: Passed to tensor_statistics()
    
    Returns:
        Formatted string if return_str=True, else None
    """
    data = convert_dataclass_to_dict(data)
    
    def _detach_to_cpu(obj: Any) -> Any:
        """Recursively detach tensors and move to CPU."""
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        elif isinstance(obj, (list, tuple)):
            return type(obj)(_detach_to_cpu(item) for item in obj)
        elif isinstance(obj, dict):
            return {k: _detach_to_cpu(v) for k, v in obj.items()}
        elif hasattr(obj, '__dict__'):
            return {k: _detach_to_cpu(v) for k, v in obj.__dict__.items()}
        else:
            return obj
    
    if save_path is not None:
        try:
            data_to_save = _detach_to_cpu(data)
            dirname = os.path.dirname(save_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            torch.save(data_to_save, save_path)
            print(f"Saved data to: {save_path}")
        except Exception as e:
            print(f"Failed to save data to {save_path}: {e}\n{traceback.format_exc()}")
    
    lines: List[str] = []
    
    try:
        data = dict(data)
    except (TypeError, ValueError):
        pass
    
    def add_line(text: str) -> None:
        if return_str:
            lines.append(text)
        else:
            print(text)
    
    def _process_nested_item(item: Any, item_prefix: str, max_str_len: int, 
                             return_str: bool, lines: List[str], **kwargs) -> None:
        sub_result = print_input_info(item, item_prefix, max_str_len, return_str=True, **kwargs)
        if return_str:
            lines.extend(sub_result.split('\n'))
        else:
            print(sub_result)
    
    if data is None:
        add_line(f"{prefix}None")
        return "\n".join(lines) if return_str else None
    
    if isinstance(data, torch.Tensor):
        flattened = data.flatten()
        data_preview = f"{flattened[:max_show].tolist()}...{flattened[-max_show:].tolist()}"
        base_info = (f"{prefix}Tensor: shape={tuple(data.shape)}, dtype={data.dtype}, "
                    f"device={data.device}, data={data_preview}")
        
        if data.dtype == torch.bool:
            total_elements = data.numel()
            true_count = data.sum().item()
            false_count = total_elements - true_count
            true_ratio = true_count / total_elements * 100 if total_elements > 0 else 0
            false_ratio = false_count / total_elements * 100 if total_elements > 0 else 0
            
            add_line(base_info)
            add_line(f"{prefix}  True:  count={true_count:,d} ({true_ratio:.2f}%)")
            add_line(f"{prefix}  False: count={false_count:,d} ({false_ratio:.2f}%)")
        else:
            add_line(base_info)
            for idx, stat_line in enumerate(tensor_statistics(data, **kwargs)):
                add_line(f"{prefix}  stat{idx}:  {stat_line}")
    
    elif isinstance(data, str):
        display_str = data[:max_str_len] + "..." if len(data) > max_str_len else data
        add_line(f"{prefix}String: length={len(data)}, value='{display_str}'")
    
    elif isinstance(data, (list, tuple)):
        container_type = "List" if isinstance(data, list) else "Tuple"
        add_line(f"{prefix}{container_type}: length={len(data)}")
        for i, item in enumerate(data):
            add_line(f"{prefix}[{i}]:")
            _process_nested_item(item, prefix + "  ", max_str_len, return_str, lines, **kwargs)
    
    elif isinstance(data, dict):
        add_line(f"{prefix}Dict: keys={len(data)}")
        for key, value in data.items():
            add_line(f"{prefix}'{key}':")
            _process_nested_item(value, prefix + "  ", max_str_len, return_str, lines, **kwargs)
    
    elif isinstance(data, (int, float)):
        add_line(f"{prefix}{type(data).__name__}: {data}")
    
    else:
        data_str = str(data)
        truncated = data_str[:max_show] + "..." + data_str[-max_show:] if len(data_str) > max_show * 2 else data_str
        add_line(f"{prefix}Other type ({type(data).__name__}): {truncated}")
    
    return "\n".join(lines) if return_str else None


def format_dict_or_list(obj: Any, indent_level: int = 0, indent_size: int = 2) -> str:
    """Format dict/list as readable string (alternative to json.dumps).
    
    Args:
        obj: Dictionary, list, or other object
        indent_level: Current indentation level
        indent_size: Spaces per indentation level
    
    Returns:
        Formatted string
    """
    def format_value(value: Any, indent_level: int, indent_size: int) -> str:
        if isinstance(value, (dict, list)):
            return format_dict_or_list(value, indent_level, indent_size)
        elif isinstance(value, str):
            return f'"{value}"'
        else:
            return str(value)
    
    if isinstance(obj, dict):
        formatted_items = []
        indent = " " * indent_size * (indent_level + 1)
        for key, value in obj.items():
            formatted_value = format_value(value, indent_level + 1, indent_size)
            formatted_items.append(f'{indent}"{key}": {formatted_value}')
        
        items_str = ',\n'.join(formatted_items)
        current_indent = " " * indent_size * indent_level
        return f'{{\n{items_str}\n{current_indent}}}'
    
    elif isinstance(obj, list):
        formatted_items = []
        indent = " " * indent_size * (indent_level + 1)
        for item in obj:
            formatted_value = format_value(item, indent_level + 1, indent_size)
            formatted_items.append(f'{indent}{formatted_value}')
        
        items_str = ',\n'.join(formatted_items)
        current_indent = " " * indent_size * indent_level
        return f'[\n{items_str}\n{current_indent}]'
    
    else:
        return str(obj)