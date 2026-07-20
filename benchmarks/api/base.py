"""
Base LLM Client Definition
Provides unified interface specification with retry mechanism and batch processing
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed


class BaseLLMClient(ABC):
    """
    Base class for LLM clients, defining unified interface

    All concrete LLM clients (Gemini, DeepSeek, etc.) should inherit from this class
    Provides unified retry mechanism and batch processing capabilities
    """

    def __init__(self, **config):
        """
        Initialize client

        Args:
            **config: Model-specific configuration parameters
        """
        self.config = config
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 2)
        self._setup()

    @abstractmethod
    def _setup(self):
        """Setup client (subclasses implement specific initialization logic)"""
        pass

    @abstractmethod
    def _call_api(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Call API to generate text (subclasses implement specific API call logic)

        Args:
            prompt: Input prompt
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens to generate
            **kwargs: Other model-specific parameters

        Returns:
            Generated text content

        Raises:
            Exception: Raised when API call fails
        """
        pass

    def _is_retryable_error(self, error_msg: str) -> bool:
        """
        Determine if error is retryable

        Args:
            error_msg: Error message

        Returns:
            bool: Whether the error is retryable
        """
        retryable_keywords = [
            '503', '429', '500', 'timeout', 'timed out', 'deadline',
            'unavailable', 'failed to connect', 'connection',
            'rate limit', 'overload'
        ]
        return any(keyword in error_msg.lower() for keyword in retryable_keywords)

    def _generate_with_retry(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generation method with retry mechanism (template method)

        Args:
            prompt: Input prompt
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens to generate
            **kwargs: Other parameters

        Returns:
            str: Generated text content

        Raises:
            Exception: Raised when API call fails
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt cannot be empty")

        last_error = None

        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    jitter = random.uniform(0, delay * 0.3)
                    time.sleep(delay + jitter)

                return self._call_api(prompt, temperature, max_tokens, **kwargs)

            except Exception as e:
                last_error = e
                error_msg = str(e)

                is_retryable = self._is_retryable_error(error_msg)

                if attempt == self.max_retries - 1 or not is_retryable:
                    raise Exception(f"{self.__class__.__name__} API call failed: {error_msg}")

                print(f"{self.__class__.__name__} API call failed "
                      f"(attempt {attempt + 1}/{self.max_retries}), "
                      f"will retry in {self.retry_delay} seconds: {error_msg[:100]}")

        raise Exception(f"Maximum retry attempts reached ({self.max_retries}): {last_error}")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text content (public interface)

        Args:
            prompt: Input prompt
            temperature: Temperature parameter (controls randomness)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Other model-specific parameters

        Returns:
            str: Generated text content

        Raises:
            ValueError: Parameter error
            Exception: API call failed
        """
        return self._generate_with_retry(prompt, temperature, max_tokens, **kwargs)

    def batch_generate(
        self,
        prompts: List[str],
        max_workers: int = 5,
        show_progress: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Batch generate text (with concurrent support)

        Args:
            prompts: List of prompts
            max_workers: Maximum number of concurrent threads, default 5
            show_progress: Whether to show progress bar, default True
            **kwargs: Other parameters to pass to generate

        Returns:
            List[Dict]: List of results, each element contains:
                - prompt: Original prompt
                - result: Generated text (on success)
                - error: Error message (on failure)
                - success: Whether successful
        """
        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False
            if show_progress:
                print("Warning: tqdm not installed, cannot show progress bar")

        def process_prompt(prompt: str, index: int) -> Dict[str, Any]:
            try:
                result = self.generate(prompt, **kwargs)
                return {
                    "index": index,
                    "prompt": prompt,
                    "result": result,
                    "success": True
                }
            except Exception as e:
                return {
                    "index": index,
                    "prompt": prompt,
                    "error": str(e),
                    "success": False
                }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(process_prompt, prompt, i): i
                for i, prompt in enumerate(prompts)
            }
            if show_progress and has_tqdm:
                progress = tqdm(
                    as_completed(future_to_index),
                    total=len(prompts),
                    desc=f"Generating ({self.__class__.__name__})"
                )
            else:
                progress = as_completed(future_to_index)

            temp_results = []
            for future in progress:
                try:
                    result = future.result()
                    temp_results.append(result)
                except Exception as e:
                    index = future_to_index[future]
                    temp_results.append({
                        "index": index,
                        "prompt": prompts[index],
                        "error": f"Task execution failed: {str(e)}",
                        "success": False
                    })

        results = sorted(temp_results, key=lambda x: x["index"])

        for r in results:
            r.pop("index", None)

        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
