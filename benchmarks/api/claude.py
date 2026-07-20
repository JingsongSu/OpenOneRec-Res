"""
Claude API Client Implementation
Based on Anthropic official SDK
"""
from typing import Optional
from anthropic import Anthropic
from .base import BaseLLMClient


class ClaudeClient(BaseLLMClient):
    """
    Claude API Client

    Example:
        >>> client = ClaudeClient(
        ...     api_key="your-api-key",
        ...     model_name="claude-sonnet-4-20250514"
        ... )
        >>> response = client.generate("Tell me a joke")
    """

    def _setup(self):
        """Initialize Claude client"""
        self.api_key = self.config.get("api_key")
        self.model_name = self.config.get("model_name", "claude-sonnet-4-20250514")
        self.base_url = self.config.get("base_url")
        self.default_max_tokens = self.config.get("max_new_tokens", 1024)
        self.default_temperature = self.config.get("temperature", 1.0)

        if not self.api_key:
            raise ValueError("api_key is a required parameter")

        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = Anthropic(**client_kwargs)

    def _call_api(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Call Claude API to generate text

        Args:
            prompt: Input prompt
            temperature: Temperature parameter (0.0-1.0), default 1.0
            max_tokens: Maximum number of tokens to generate, default 1024
            **kwargs: Other Claude-specific parameters, such as:
                - system: System prompt
                - top_p: Nucleus sampling parameter
                - top_k: Top-k sampling parameter

        Returns:
            str: Generated text content

        Raises:
            Exception: Raised when API call fails
        """
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        system = kwargs.pop("system", None)

        request_params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }

        if temperature is not None:
            request_params["temperature"] = temperature
        if system:
            request_params["system"] = system

        for key in ["top_p", "top_k", "stop_sequences"]:
            if key in kwargs:
                request_params[key] = kwargs.pop(key)

        response = self.client.messages.create(**request_params)

        if response and response.content:
            text_blocks = [
                block.text for block in response.content
                if hasattr(block, 'text')
            ]
            if text_blocks:
                return "".join(text_blocks)
            else:
                raise Exception("API returned empty response")
        else:
            raise Exception("API returned invalid response")
