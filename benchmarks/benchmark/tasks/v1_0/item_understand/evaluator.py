"""
Item Understand Evaluator

Evaluates model predictions on Item Understand task using WIP (LLM-as-Judge).
"""

import os
from typing import Dict, Any, Tuple, List

from benchmark.console import console
from benchmark.tasks.v1_0.base_evaluator import BaseEval


class ItemUnderstandEvaluator(BaseEval):
    """Item Understand task evaluator"""

    @property
    def required_metrics(self) -> List[str]:
        """Define required overall metrics for Item Understand evaluation"""
        return ["macro_wip_double_weighted_f1"]


    def _compute_metrics_from_scratch(self) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Compute all metrics from scratch

        Returns:
            Tuple of (metrics, per_sample_metrics)
        """
        total_samples = len(self.samples)

        # Prepare data for evaluation
        sample_ids = list(self.samples.keys())
        predictions = []
        references = []

        for sample_id in sample_ids:
            sample = self.samples[sample_id]

            # Get ground truth
            ground_truth = sample.get("ground_truth", "")
            references.append(ground_truth)

            # Get model prediction (first generation)
            generations = sample.get("generations", [])
            if not generations:
                prediction = ""
            else:
                prediction = generations[0]
            predictions.append(prediction)

        # Get evaluation config
        eval_config = self.task_config.get("evaluation_config", {})

        # Build per-sample metrics
        per_sample_metrics = {}
        for sample_id in sample_ids:
            per_sample_metrics[sample_id] = {}

        # Build overall metrics
        metrics = {
            "num_samples": total_samples
        }

        # WIP Evaluation (if enabled)
        wip_enabled = eval_config.get("wip_enabled", False)
        if wip_enabled:
            console.print("[cyan]WIP evaluation enabled, starting LLM-as-Judge evaluation...[/cyan]")
            wip_metrics, wip_per_sample = self._evaluate_wip(
                sample_ids=sample_ids,
                predictions=predictions,
                references=references,
                eval_config=eval_config
            )

            # Merge WIP metrics into overall metrics
            metrics.update(wip_metrics)

            # Merge WIP per-sample metrics
            for sample_id in sample_ids:
                if sample_id in wip_per_sample:
                    per_sample_metrics[sample_id].update(wip_per_sample[sample_id])

        # Save debug information if requested
        if self.debug and self.predictions_dir:
            self._save_debug_info(metrics, per_sample_metrics, predictions, references)

        return metrics, per_sample_metrics

    def _evaluate_wip(
        self,
        sample_ids: list,
        predictions: list,
        references: list,
        eval_config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Perform WIP (Weighted Information Points) evaluation using LLM-as-Judge.

        Args:
            sample_ids: List of sample IDs
            predictions: List of prediction texts
            references: List of reference texts
            eval_config: Evaluation configuration

        Returns:
            Tuple of (wip_metrics, wip_per_sample_metrics)
        """
        try:
            from api import get_client_from_config
            from benchmark.tasks.v1_0.item_understand.utils import evaluate_wip
        except ImportError as e:
            console.print(f"[red]Failed to import WIP evaluation modules: {e}[/red]")
            return {}, {}

        # Get WIP config
        wip_judge_model = eval_config.get("wip_judge_model", "deepseek")
        wip_max_workers = eval_config.get("wip_max_workers", 5)
        wip_max_samples = eval_config.get("wip_max_samples", 100)
        wip_core_threshold = eval_config.get("wip_core_threshold", 5)
        wip_gt_cache_dir = os.path.join(self.data_dir, self.task_name)  # Use data_dir / task_name as GT cache directory

        # Use BERTScore config from evaluation_config (not separate wip config)
        bertscore_model = eval_config.get("bertscore_model_type", "bert-base-chinese")
        bertscore_num_layers = eval_config.get("bertscore_num_layers", 9)

        # Create LLM client
        try:
            llm_client = get_client_from_config(wip_judge_model)
            console.print(f"[green]Using {wip_judge_model} as WIP judge[/green]")
        except Exception as e:
            console.print(f"[red]Failed to create LLM client for WIP evaluation: {e}[/red]")
            return {}, {}

        # Prepare data as dicts
        predictions_dict = {id: pred for id, pred in zip(sample_ids, predictions)}
        references_dict = {id: ref for id, ref in zip(sample_ids, references)}

        # Get model name for cache file naming
        # Try to extract from llm_client config
        model_name = getattr(llm_client, 'model_name', wip_judge_model)

        # Run WIP evaluation
        try:
            wip_metrics, wip_per_sample = evaluate_wip(
                predictions=predictions_dict,
                references=references_dict,
                llm_client=llm_client,
                max_workers=wip_max_workers,
                max_samples=wip_max_samples,
                gt_cache_dir=wip_gt_cache_dir,
                model_name=model_name,
                save_dir=self.predictions_dir,
                bertscore_model=bertscore_model,
                bertscore_num_layers=bertscore_num_layers,
                core_threshold=wip_core_threshold,
            )

            console.print(f"[green]WIP evaluation completed: {wip_metrics.get('wip_num_samples', 0)} samples evaluated[/green]")
            return wip_metrics, wip_per_sample

        except Exception as e:
            console.print(f"[red]WIP evaluation failed: {e}[/red]")
            import traceback
            traceback.print_exc()
            return {}, {}

    def _save_debug_info(
        self,
        metrics: Dict[str, Any],
        per_sample_metrics: Dict[str, Dict[str, Any]],
        predictions: list,
        references: list
    ):
        """
        Save detailed debug information to file

        Args:
            metrics: Overall metrics
            per_sample_metrics: Per-sample metrics
            predictions: List of predictions
            references: List of references
        """
        # Prepare debug info
        debug_info = {
            "overall_metrics": metrics,
            "per_sample_metrics": per_sample_metrics,
            "sample_count": len(predictions),
        }

        # Add some examples
        sample_ids = list(self.samples.keys())
        debug_info["examples"] = []
        for i in range(min(10, len(sample_ids))):
            sample_id = sample_ids[i]
            debug_info["examples"].append({
                "sample_id": sample_id,
                "prediction": predictions[i],
                "reference": references[i],
                "wip_unweighted_f1": per_sample_metrics[sample_id].get("wip_unweighted_f1"),
                "wip_unweighted_core_f1": per_sample_metrics[sample_id].get("wip_unweighted_core_f1"),
                "wip_importance_weighted_f1": per_sample_metrics[sample_id].get("wip_importance_weighted_f1"),
                "wip_importance_weighted_core_f1": per_sample_metrics[sample_id].get("wip_importance_weighted_core_f1"),
                "wip_double_weighted_f1": per_sample_metrics[sample_id].get("wip_double_weighted_f1"),
                "wip_double_weighted_core_f1": per_sample_metrics[sample_id].get("wip_double_weighted_core_f1"),
            })

        # Save to file using base class method
        self._save_debug_json(debug_info, filename="debug.json")

        # Print summary statistics
        console.print(f"Total samples: {metrics['num_samples']}")

        # Print WIP metrics if available
        if metrics.get('macro_wip_unweighted_f1') is not None:
            console.print(f"Macro WIP Unweighted F1: {metrics['macro_wip_unweighted_f1']:.4f}")
        if metrics.get('macro_wip_double_weighted_f1') is not None:
            console.print(f"Macro WIP Double-weighted F1: {metrics['macro_wip_double_weighted_f1']:.4f}")
