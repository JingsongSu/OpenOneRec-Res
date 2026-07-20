"""
Unified LLM API Wrapper
Supports convenient calling of Gemini, DeepSeek, and Claude models
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base import BaseLLMClient
from .gemini import GeminiClient
from .deepseek import DeepSeekClient
from .claude import ClaudeClient


# Model mapping
MODEL_CLASSES = {
    "gemini": GeminiClient,
    "deepseek": DeepSeekClient,
    "claude": ClaudeClient,
}


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from JSON file

    Args:
        config_path: Configuration file path, defaults to api/config/llm_config.json

    Returns:
        dict: Configuration dictionary

    Raises:
        FileNotFoundError: Configuration file does not exist
        json.JSONDecodeError: Configuration file format error
    """
    if config_path is None:
        current_dir = Path(__file__).parent
        config_path = current_dir / "config" / "llm_config.json"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_client(model: str, **config) -> BaseLLMClient:
    """
    Factory function: Create LLM client instance

    Args:
        model: Model name ("gemini" or "deepseek")
        **config: Model-specific configuration parameters

    Returns:
        BaseLLMClient: Client instance

    Raises:
        ValueError: Unsupported model type

    Example:
        >>> client = get_client("gemini",
        ...                    project="your-project",
        ...                    location="us-central1")
        >>> result = client.generate("Tell me a joke")
    """
    model = model.lower()
    if model not in MODEL_CLASSES:
        raise ValueError(
            f"Unsupported model: {model}. "
            f"Supported models: {', '.join(MODEL_CLASSES.keys())}"
        )

    client_class = MODEL_CLASSES[model]
    return client_class(**config)


def get_client_from_config(
    model: str,
    config_path: Optional[str] = None
) -> BaseLLMClient:
    """
    Create LLM client from configuration file

    Args:
        model: Model name ("gemini" or "deepseek")
        config_path: Configuration file path, defaults to api/config/llm_config.json

    Returns:
        BaseLLMClient: Client instance

    Raises:
        ValueError: Model configuration not found in configuration file

    Example:
        >>> client = get_client_from_config("gemini")
        >>> result = client.generate("Tell me a joke")
    """
    config = load_config(config_path)
    model = model.lower()

    if model not in config:
        raise ValueError(
            f"Model '{model}' configuration not found in configuration file. "
            f"Available models: {', '.join(config.keys())}"
        )

    model_config = config[model]
    return get_client(model, **model_config)


def batch_generate(
    prompts: List[str],
    model: str,
    max_workers: int = 5,
    show_progress: bool = True,
    config_path: Optional[str] = None,
    **config
) -> List[Dict[str, Any]]:
    """
    Batch generate text (with concurrent support)

    Args:
        prompts: List of prompts
        model: Model name ("gemini" or "deepseek")
        max_workers: Maximum number of concurrent threads, default 5
        show_progress: Whether to show progress bar, default True
        config_path: Configuration file path (if provided, use configuration file first)
        **config: Model configuration parameters (if not using configuration file)

    Returns:
        List[Dict]: List of results, each element contains:
            - prompt: Original prompt
            - result: Generated text (on success)
            - error: Error message (on failure)
            - success: Whether successful

    Example:
        >>> # Using configuration file
        >>> results = batch_generate(
        ...     prompts=["Question 1", "Question 2", "Question 3"],
        ...     model="gemini",
        ...     max_workers=3
        ... )

        >>> # Direct configuration
        >>> results = batch_generate(
        ...     prompts=["Question 1", "Question 2"],
        ...     model="deepseek",
        ...     api_key="your-key",
        ...     appid="your-appid"
        ... )
    """
    if config_path:
        client = get_client_from_config(model, config_path)
    else:
        client = get_client(model, **config)

    return client.batch_generate(
        prompts=prompts,
        max_workers=max_workers,
        show_progress=show_progress
    )


# Export all public interfaces
__all__ = [
    # Classes
    "BaseLLMClient",
    "GeminiClient",
    "DeepSeekClient",
    "ClaudeClient",
    # Functions
    "get_client",
    "get_client_from_config",
    "batch_generate",
    "load_config",
]
