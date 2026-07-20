"""
Base Evaluator for all task evaluators

Provides common interface for evaluation logic.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
from benchmark.console import console, success_style


class BaseEval(ABC):
    """Base class for all task evaluators"""

    def __init__(
        self,
        samples: Dict[str, Dict[str, Any]],
        task_name: Optional[str] = None,
        predictions_dir: Optional[str] = None,
        debug: bool = False,
        task_config: Optional[Dict[str, Any]] = None,
        data_dir: Optional[str] = None,
        overwrite: bool = False,
        cached_metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize base evaluator

        Args:
            samples: Dictionary of samples from test_generated.json
                Format: {
                    sample_id: {
                        "prompt": "...",
                        "generations": ["..."],
                        "ground_truth": "...",
                        "metadata": {...}
                    }
                }
            task_name: Task name (e.g., "math_500")
            predictions_dir: Directory to save debug files (optional)
            debug: Whether to save debug information
            task_config: Task configuration dictionary (optional)
            data_dir: Data directory path (optional)
            overwrite: Whether to overwrite existing metrics and recompute from scratch
            cached_metrics: Existing overall metrics from eval_results (optional)
        """
        self.samples = samples
        self.task_name = task_name
        self.predictions_dir = predictions_dir
        self.debug = debug
        self.task_config = task_config or {}
        self.data_dir = data_dir
        self.overwrite = overwrite
        self.cached_metrics = cached_metrics or {}
    
    def evaluate(self) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Evaluate the samples and return metrics

        This method provides a simplified two-level caching-aware evaluation flow:
        1. If overwrite=True, always recompute from scratch
        2. If cached overall metrics exist in eval_results, return them with empty per_sample_metrics
        3. Otherwise, compute from scratch

        Subclasses should override:
        - required_metrics property: Return list of overall metric names
        - _compute_metrics_from_scratch(): Compute all metrics from scratch

        Returns:
            Tuple of (metrics, per_sample_metrics)
        """

        # If overwrite=True, always recompute from scratch
        if self.overwrite:
            console.print("[cyan]Overwrite=True, recomputing all metrics from scratch...[/cyan]")
            return self._compute_metrics_from_scratch()

        # If cached overall metrics exist, use them
        if self._has_all_required_metrics():
            console.print("[cyan]Using existing overall metrics from eval_results...[/cyan]")
            # Return cached metrics with empty per_sample_metrics (not needed when using cache)
            return self.cached_metrics, {}

        # Otherwise, compute from scratch
        console.print("[cyan]Computing metrics from scratch...[/cyan]")
        return self._compute_metrics_from_scratch()

    def _all_samples_have_keys(self, required_keys: List[str]) -> bool:
        """Check if all samples have required keys"""
        for sample in self.samples.values():
            for key in required_keys:
                if key not in sample:
                    return False
        return True

    @property
    def required_metrics(self) -> Optional[List[str]]:
        """Define required overall metric keys"""
        return None

    def _has_all_required_metrics(self) -> bool:
        """Check if cached_metrics contains all required keys (override for custom logic)"""
        if self.required_metrics is not None:
            return all(key in self.cached_metrics for key in self.required_metrics)
        return False

    def _compute_metrics_from_scratch(self) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Compute metrics from scratch (override in subclasses)"""
        raise NotImplementedError("Subclasses must implement _compute_metrics_from_scratch()")


    def _save_debug_json(
        self,
        debug_info: Dict[str, Any],
        filename: str = "debug.json"
    ) -> Optional[str]:
        """Save debug information to JSON file"""
        if not self.predictions_dir:
            return None

        debug_filename = os.path.join(self.predictions_dir, filename)
        os.makedirs(os.path.dirname(debug_filename), exist_ok=True)

        with open(debug_filename, 'w', encoding='utf-8') as f:
            json.dump(debug_info, f, indent=2, ensure_ascii=False)
            
        console.print(f"✓ Debug information saved to: {debug_filename}", style=success_style)
        return debug_filename
