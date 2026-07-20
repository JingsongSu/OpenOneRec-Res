"""
Label Prediction Task Module

Classification task for predicting user engagement with video content.
Uses logprobs-based classification with AUC and wuAUC metrics.
"""

from .config import LABEL_PRED_CONFIG
from .evaluator import LabelPredEvaluator
from . import utils

__all__ = [
    "LABEL_PRED_CONFIG",
    "LabelPredEvaluator",
    "utils",
]

