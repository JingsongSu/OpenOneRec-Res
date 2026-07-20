"""
Label Prediction Task Evaluator

Evaluator for label_pred classification task.
Computes AUC metric from logprobs-based predictions.
"""

from typing import Dict, Any, Tuple, List

from benchmark.console import console
from benchmark.tasks.v1_0.base_evaluator import BaseEval
from benchmark.tasks.v1_0.label_pred.utils import (
    extract_label_from_answer,
    extract_probability_from_logprobs,
    calculate_auc,
    get_debug_info,
)


class LabelPredEvaluator(BaseEval):
    """
    Label prediction task evaluator

    This is a classification task for predicting user engagement.
    Uses logprobs-based predictions to compute AUC metric.

    Metrics:
    - AUC: Area Under ROC Curve
    """

    @property
    def required_metrics(self) -> List[str]:
        """Define required overall metrics for label prediction evaluation"""
        return ["auc"]

    def _compute_metrics_from_scratch(self) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Compute all evaluation metrics from scratch

        Extracts probabilities from logprobs and computes AUC metric.
        Also stores per-sample metrics back into self.samples for caching.

        Returns:
            Tuple of (metrics, per_sample_metrics):
            - metrics: Overall metrics including auc, etc.
            - per_sample_metrics: Per-sample evaluation results
        """
        total_samples = len(self.samples)

        # Extract predictions and labels
        predictions = {}  # {sample_id: probability}
        labels = {}  # {sample_id: 0 or 1}
        
        # Per-sample metrics
        per_sample_metrics = {}
        
        # Debug information collection
        debug_info = {
            "correct_predictions": [],
            "incorrect_predictions": [],
            "invalid_samples": [],
        }
        
        for sample_id, sample in self.samples.items():
            # Get ground truth answer
            ground_truth = sample.get("ground_truth", "")
            
            # Extract label from ground truth
            label = extract_label_from_answer(ground_truth)
            
            if label == -1:
                # Invalid label
                console.print(f"[yellow]Sample {sample_id}: unrecognized answer '{ground_truth}'[/yellow]")
                if self.debug:
                    debug_info["invalid_samples"].append({
                        "sample_id": sample_id,
                        "ground_truth": ground_truth,
                        "reason": "unrecognized_label"
                    })
                continue
            
            labels[sample_id] = label

            # Get model prediction (logprobs dictionary)
            # For label_pred, generations contains {token: probability} dict
            generations = sample.get("generations", {})

            # Variables to store probability extraction results
            predicted_prob = 0.5
            parsed_probs = None
            normalized_probs = None

            if not generations:
                # No generation - log as invalid sample
                console.print(f"[yellow]Sample {sample_id}: no generation found[/yellow]")
                if self.debug:
                    debug_info["invalid_samples"].append({
                        "sample_id": sample_id,
                        "ground_truth": ground_truth,
                        "reason": "no_generation"
                    })
                # Skip this sample - don't include in predictions
                continue
            else:
                try:
                    # Extract probability for positive class ("是")
                    # Now returns dict with parsed_probs, normalized_probs, and score
                    prob_result = extract_probability_from_logprobs(
                        generations,
                        positive_token="是",
                        negative_token="否",
                        sample_id=sample_id
                    )

                    predicted_prob = prob_result["score"]
                    parsed_probs = prob_result["parsed_probs"]
                    normalized_probs = prob_result["normalized_probs"]

                except ValueError as e:
                    # Parsing failed - log detailed error and skip sample
                    console.print(f"[red]Sample {sample_id}: {str(e)}[/red]")
                    if self.debug:
                        debug_info["invalid_samples"].append({
                            "sample_id": sample_id,
                            "ground_truth": ground_truth,
                            "reason": "parsing_error",
                            "error": str(e)
                        })
                    # Skip this sample - don't include in predictions
                    continue

            predictions[sample_id] = predicted_prob

            # Store per-sample metrics (both in return dict and in self.samples for caching)
            sample_metrics = {
                "label": label,
                "predicted_prob": predicted_prob,
            }
            per_sample_metrics[sample_id] = sample_metrics

            # Cache metrics in self.samples for future use, including debug info
            self.samples[sample_id]["label"] = label
            self.samples[sample_id]["predicted_prob"] = predicted_prob

            # Add new debug fields to sample for tracking
            self.samples[sample_id]["y_true"] = label
            self.samples[sample_id]["y_score"] = predicted_prob
            if parsed_probs is not None:
                self.samples[sample_id]["parsed_probs"] = parsed_probs
            if normalized_probs is not None:
                self.samples[sample_id]["normalized_probs"] = normalized_probs

            # Debug information collection
            if self.debug:
                debug_item = get_debug_info(
                    sample_id=sample_id,
                    logprobs_dict=parsed_probs,
                    predicted_prob=predicted_prob,
                    ground_truth=ground_truth,
                    label=label,
                )
                # Determine if prediction is correct
                # Correct: (predicted_prob > 0.5 and label = 1) OR (predicted_prob <= 0.5 and label = 0)
                is_correct = (predicted_prob > 0.5 and label == 1) or (predicted_prob <= 0.5 and label == 0)

                if is_correct:
                    debug_info["correct_predictions"].append(debug_item)
                else:
                    debug_info["incorrect_predictions"].append(debug_item)
        
        # Calculate AUC
        auc = calculate_auc(predictions, labels)

        # Prepare overall metrics
        metrics = {
            "auc": auc,
            "total_samples": total_samples,
            "valid_samples": len(labels),
            "invalid_samples": len(debug_info["invalid_samples"]) if self.debug else 0,
        }

        # Save debug information if requested
        if self.debug and self.predictions_dir:
            self._save_debug_info(debug_info, metrics)

        return metrics, per_sample_metrics

    def _save_debug_info(
        self,
        debug_info: Dict[str, Any],
        metrics: Dict[str, Any],
    ):
        """
        Save detailed debug information to file

        Args:
            debug_info: Debug information dictionary
            metrics: Overall metrics
        """
        # Add statistics to debug_info
        debug_info["statistics"] = {
            "total_samples": metrics["total_samples"],
            "valid_samples": metrics["valid_samples"],
            "correct_predictions_count": len(debug_info["correct_predictions"]),
            "incorrect_predictions_count": len(debug_info["incorrect_predictions"]),
            "invalid_samples_count": len(debug_info["invalid_samples"]),
        }

        # Add metrics
        debug_info["metrics"] = metrics

        # Save debug info to file using base class method
        self._save_debug_json(debug_info, filename="debug.json")

        console.print(f"Total samples: {metrics['total_samples']}")
        console.print(f"Valid samples: {metrics['valid_samples']}")
        console.print(f"Correct predictions: {len(debug_info['correct_predictions'])}")
        console.print(f"Incorrect predictions: {len(debug_info['incorrect_predictions'])}")

        # Calculate and display accuracy if we have valid predictions
        total_predictions = len(debug_info['correct_predictions']) + len(debug_info['incorrect_predictions'])
        if total_predictions > 0:
            accuracy = len(debug_info['correct_predictions']) / total_predictions * 100
            console.print(f"Accuracy: {accuracy:.2f}%")

        console.print(f"Invalid samples: {len(debug_info['invalid_samples'])}")

        # Print metrics
        console.print("\n[bold]Metrics:[/bold]")
        console.print(f"  AUC: {metrics['auc']:.4f}")

        # Show some invalid sample examples
        if debug_info["invalid_samples"]:
            console.print(f"\n[yellow]Invalid sample examples (first 3):[/yellow]")
            for i, item in enumerate(debug_info["invalid_samples"][:3]):
                console.print(f"  Example {i+1}:")
                console.print(f"    Sample ID: {item['sample_id']}")
                console.print(f"    Reason: {item['reason']}")
                console.print(f"    Ground truth: {item['ground_truth']}")
                console.print()

