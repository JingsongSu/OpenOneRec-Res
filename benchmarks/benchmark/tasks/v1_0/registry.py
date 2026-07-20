"""
Task Registry - Unified Task Registration

This module consolidates:
- loader_factory.py
- evaluator_factory.py  
- tasks.py

Purpose: Each task is defined in ONE place only, avoiding duplication across multiple files.
"""

from dataclasses import dataclass
from typing import Type, Dict, Any, Optional

# ===== Import all configs =====
from .label_pred.config import LABEL_PRED_CONFIG
from .item_understand.config import ITEM_UNDERSTAND_CONFIG
from .rec_reason.config import REC_REASON_CONFIG
from .recommendation.config import (
    LABEL_COND_CONFIG,
    VIDEO_CONFIG,
    PRODUCT_CONFIG,
    AD_CONFIG,
    INTERACTIVE_CONFIG,
)

# ===== Import base loader =====
from .base_loader import BaseLoader

# ===== Import all evaluators =====
from .label_pred.evaluator import LabelPredEvaluator
from .item_understand.evaluator import ItemUnderstandEvaluator
from .rec_reason.evaluator import RecoReasonEvaluator
from .recommendation.evaluator import RecommendationEvaluator


@dataclass
class TaskRegistration:
    """Task registration information"""
    name: str
    config: Dict[str, Any]
    evaluator_class: Type
    category: str  # "general", "recommendation", "caption"


# ========================================
# Unified Task Registry
# ========================================
TASK_REGISTRY: Dict[str, TaskRegistration] = {
    "label_cond": TaskRegistration(
        name="label_cond",
        config=LABEL_COND_CONFIG,
        evaluator_class=RecommendationEvaluator,
        category="recommendation"
    ),
    "video": TaskRegistration(
        name="video",
        config=VIDEO_CONFIG,
        evaluator_class=RecommendationEvaluator,
        category="recommendation"
    ),
    "product": TaskRegistration(
        name="product",
        config=PRODUCT_CONFIG,
        evaluator_class=RecommendationEvaluator,
        category="recommendation"
    ),
    "ad": TaskRegistration(
        name="ad",
        config=AD_CONFIG,
        evaluator_class=RecommendationEvaluator,
        category="recommendation"
    ),
    "interactive": TaskRegistration(
        name="interactive",
        config=INTERACTIVE_CONFIG,
        evaluator_class=RecommendationEvaluator,
        category="recommendation"
    ),
    "label_pred": TaskRegistration(
        name="label_pred",
        config=LABEL_PRED_CONFIG,
        evaluator_class=LabelPredEvaluator,
        category="recommendation"
    ),
    "item_understand": TaskRegistration(
        name="item_understand",
        config=ITEM_UNDERSTAND_CONFIG,
        evaluator_class=ItemUnderstandEvaluator,
        category="caption"
    ),
    "rec_reason": TaskRegistration(
        name="rec_reason",
        config=REC_REASON_CONFIG,
        evaluator_class=RecoReasonEvaluator,
        category="caption"
    ),
}


# ========================================
# Factory Functions
# ========================================

def get_loader(task_name: str, data_dir: str, tokenizer: Optional[Any] = None, enable_thinking: Optional[bool] = None):
    """
    Get loader instance for a task

    Replaces loader_factory.get_loader()

    Args:
        task_name: Name of the task
        benchmark_version: Version of the benchmark (used for task selection, not passed to loader)
        data_dir: Data directory path
        tokenizer: Tokenizer instance (optional, required for message-based formats)
        enable_thinking: Enable thinking mode (optional, overrides task config if set)

    Returns:
        Loader instance

    Raises:
        ValueError: If task_name is not registered
    """
    if task_name not in TASK_REGISTRY:
        available_tasks = ", ".join(TASK_REGISTRY.keys())
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Available tasks: {available_tasks}"
        )

    reg = TASK_REGISTRY[task_name]

    # Create loader instance with aligned parameters
    return BaseLoader(
        task_config=reg.config,
        data_dir=data_dir,
        tokenizer=tokenizer,
        enable_thinking=enable_thinking
    )


def get_evaluator(task_name: str):
    """
    Get evaluator class for a task
    
    Replaces evaluator_factory.get_evaluator()
    
    Args:
        task_name: Name of the task
        
    Returns:
        Evaluator class (not instance)
        
    Raises:
        ValueError: If task_name is not registered
    """
    if task_name not in TASK_REGISTRY:
        available_tasks = ", ".join(TASK_REGISTRY.keys())
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Available tasks: {available_tasks}"
        )
    
    return TASK_REGISTRY[task_name].evaluator_class


def get_task_config(task_name: str) -> Dict[str, Any]:
    """
    Get task configuration
    
    Args:
        task_name: Name of the task
        
    Returns:
        Task configuration dictionary
        
    Raises:
        ValueError: If task_name is not registered
    """
    if task_name not in TASK_REGISTRY:
        available_tasks = ", ".join(TASK_REGISTRY.keys())
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Available tasks: {available_tasks}"
        )
    
    return TASK_REGISTRY[task_name].config


def get_all_tasks() -> list:
    """
    Get list of all registered task names
    
    Returns:
        List of task names
    """
    return list(TASK_REGISTRY.keys())


def get_tasks_by_category(category: str) -> list:
    """
    Get tasks filtered by category
    
    Args:
        category: Category name ("general", "recommendation", "caption")
        
    Returns:
        List of task names in the specified category
    """
    return [
        name for name, reg in TASK_REGISTRY.items()
        if reg.category == category
    ]

# ========================================
# Backward Compatibility
# ========================================

# Replaces tasks.py - TaskTable
TaskTable = {name: reg.config for name, reg in TASK_REGISTRY.items()}