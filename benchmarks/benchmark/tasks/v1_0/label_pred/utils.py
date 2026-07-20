"""
Label Prediction Task Utilities

Functions for label extraction, probability processing, and AUC/wuAUC computation.
"""

import json
import numpy as np
from typing import Dict, Tuple, Any, List
from sklearn.metrics import roc_auc_score
from benchmark.console import console


def extract_label_from_answer(answer: str) -> int:
    """
    Extract binary label from answer string
    
    Args:
        answer: Answer string (e.g., "是<|im_end|>" or "否<|im_end|>")
    
    Returns:
        1 if positive ("是"), 0 if negative ("否"), -1 if unrecognized
    
    Examples:
        >>> extract_label_from_answer("是<|im_end|>")
        1
        >>> extract_label_from_answer("否")
        0
    """
    if "是" in answer:
        return 1
    elif "否" in answer:
        return 0
    else:
        return -1


def extract_probability_from_logprobs(
    generations: List[str],
    positive_token: str = "是",
    negative_token: str = "否",
    sample_id: str = None
) -> Dict[str, Any]:
    """
    Extract probability for positive class from generations list containing reasoning and JSON probabilities.
    Applies softmax normalization to ensure probabilities sum to 1.

    Args:
        generations: List of strings, each containing reasoning text followed by JSON probabilities
        positive_token: Token representing positive class (default "是")
        negative_token: Token representing negative class (default "否")
        sample_id: Optional sample ID for error messages

    Returns:
        Dictionary containing:
        - parsed_probs: Original parsed probabilities before normalization
        - normalized_probs: Softmax normalized probabilities
        - score: Final positive class probability (after normalization)

    Raises:
        ValueError: If JSON parsing fails or required tokens are missing

    Examples:
        >>> generations = ['</think>\\n{"是": 0.7, "否": 0.3}']
        >>> result = extract_probability_from_logprobs(generations)
        >>> result['score']
        0.7
        >>> result['normalized_probs']
        {'是': 0.7, '否': 0.3}
    """
    parsed_list = []
    normalized_list = []
    scores = []

    for idx, generation in enumerate(generations):
        # Extract JSON part: check for </think> tag first
        if "</think>" in generation:
            # Extract content after </think>
            json_str = generation.split("</think>")[-1].strip()
        else:
            # No </think> tag, try to parse the entire string
            json_str = generation.strip()

        # Parse JSON and extract probability
        try:
            probs_dict = json.loads(json_str)

            # Validate that it's a dict and contains required tokens
            if not isinstance(probs_dict, dict):
                raise ValueError(f"Parsed JSON is not a dictionary: {type(probs_dict)}")

            if positive_token not in probs_dict:
                raise ValueError(f"Positive token '{positive_token}' not found in probabilities: {probs_dict}")

            if negative_token not in probs_dict:
                raise ValueError(f"Negative token '{negative_token}' not found in probabilities: {probs_dict}")

            # Extract probabilities
            p_pos = float(probs_dict[positive_token])
            p_neg = float(probs_dict[negative_token])

            # Apply softmax normalization (ensure probabilities sum to 1)
            total = p_pos + p_neg
            if total <= 0:
                raise ValueError(f"Sum of probabilities is non-positive: {total}")

            p_pos_normalized = p_pos / total
            p_neg_normalized = p_neg / total

            # Store results
            parsed_list.append({positive_token: p_pos, negative_token: p_neg})
            normalized_list.append({positive_token: p_pos_normalized, negative_token: p_neg_normalized})
            scores.append(p_pos_normalized)

        except (json.JSONDecodeError, TypeError, AttributeError, ValueError, KeyError) as e:
            # Raise detailed exception
            error_msg = f"Failed to parse generation"
            if sample_id:
                error_msg += f" for sample_id '{sample_id}'"
            error_msg += f" at index {idx}:\n"
            error_msg += f"  Generation: {generation[:200]}..." if len(generation) > 200 else f"  Generation: {generation}\n"
            error_msg += f"\n  Error: {str(e)}"
            raise ValueError(error_msg)

    # If no valid probabilities were found (empty list), raise error
    if not scores:
        error_msg = "No valid probabilities found in generations"
        if sample_id:
            error_msg += f" for sample_id '{sample_id}'"
        raise ValueError(error_msg)

    # Average across all valid elements (usually just one element)
    if len(scores) == 1:
        return {
            "parsed_probs": parsed_list[0],
            "normalized_probs": normalized_list[0],
            "score": scores[0]
        }
    else:
        # Average the probabilities for each token
        avg_parsed = {
            positive_token: sum(p[positive_token] for p in parsed_list) / len(parsed_list),
            negative_token: sum(p[negative_token] for p in parsed_list) / len(parsed_list)
        }
        avg_normalized = {
            positive_token: sum(p[positive_token] for p in normalized_list) / len(normalized_list),
            negative_token: sum(p[negative_token] for p in normalized_list) / len(normalized_list)
        }
        avg_score = sum(scores) / len(scores)

        return {
            "parsed_probs": avg_parsed,
            "normalized_probs": avg_normalized,
            "score": avg_score
        }


def calculate_auc(
    predictions: Dict[str, float],
    labels: Dict[str, int]
) -> float:
    """
    Calculate AUC (Area Under ROC Curve) using sklearn

    Args:
        predictions: Predicted probabilities, format: {sample_id: probability}
        labels: Ground truth labels (0 or 1), format: {sample_id: label}

    Returns:
        AUC value (float between 0 and 1)
    """
    if not predictions or not labels:
        console.print("[red]✗ Predictions or labels are empty[/red]")
        return 0.0

    # Align predictions and labels
    sample_ids = sorted(set(predictions.keys()) & set(labels.keys()))

    if len(sample_ids) == 0:
        console.print("[red]✗ No overlapping samples between predictions and labels[/red]")
        return 0.0

    y_true = np.array([labels[id] for id in sample_ids])
    y_scores = np.array([predictions[id] for id in sample_ids])

    # Check if we have both positive and negative samples
    if len(np.unique(y_true)) < 2:
        console.print("[yellow]⚠ Only one class present in labels, AUC is not defined[/yellow]")
        return 0.5

    try:
        # Calculate AUC using sklearn
        auc = roc_auc_score(y_true, y_scores)
        return float(auc)
    except ValueError as e:
        console.print(f"[red]✗ Error calculating AUC: {e}[/red]")
        return 0.5


def get_debug_info(
    sample_id: str,
    logprobs_dict: Dict[str, float],
    predicted_prob: float,
    ground_truth: str,
    label: int,
    user_id: str = ""
) -> Dict[str, Any]:
    """
    Prepare debug information for a sample
    
    Args:
        sample_id: Sample ID
        logprobs_dict: Dictionary of token probabilities
        predicted_prob: Predicted probability for positive class
        ground_truth: Ground truth answer string
        label: Ground truth label (0 or 1)
        user_id: User ID (optional)
    
    Returns:
        Debug information dictionary
    """
    debug_item = {
        "sample_id": sample_id,
        "ground_truth": ground_truth,
        "label": label,
        "predicted_prob": predicted_prob,
        "logprobs": logprobs_dict,
    }
    
    if user_id:
        debug_item["user_id"] = user_id
    
    return debug_item

