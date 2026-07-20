"""
DeepSeek API Client Implementation
Call DeepSeek model through Baidu Qianfan platform
"""
from typing import Optional
from openai import OpenAI
from .base import BaseLLMClient


class DeepSeekClient(BaseLLMClient):
    """
    DeepSeek API Client (through Baidu Qianfan platform)

    Example:
        >>> client = DeepSeekClient(
        ...     api_key="your-api-key",
        ...     base_url="https://qianfan.baidubce.com/v2",
        ...     model_name="deepseek-r1",
        ...     appid="your-appid"
        ... )
        >>> response = client.generate("Tell me a joke")
    """

    def _setup(self):
        """Initialize DeepSeek client"""
        self.api_key = self.config.get("api_key")
        self.base_url = self.config.get("base_url", "https://qianfan.baidubce.com/v2")
        self.model_name = self.config.get("model_name", "deepseek-r1")
        self.appid = self.config.get("appid")
        self.default_max_tokens = self.config.get("max_new_tokens", 300)
        self.default_temperature = self.config.get("temperature", 0.7)

        if not self.api_key:
            raise ValueError("api_key is a required parameter")
        if not self.appid:
            raise ValueError("appid is a required parameter")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers={"appid": self.appid}
        )

    def _call_api(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Call DeepSeek API to generate text

        Args:
            prompt: Input prompt
            temperature: Temperature parameter (0.0-2.0), default from config or 0.7
            max_tokens: Maximum number of tokens to generate, default from config or 300
            **kwargs: Other DeepSeek-specific parameters

        Returns:
            str: Generated text content

        Raises:
            Exception: Raised when API call fails
        """
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        request_params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        request_params.update(kwargs)

        response = self.client.chat.completions.create(**request_params)

        if response and response.choices:
            content = response.choices[0].message.content
            if content:
                return content
            else:
                raise Exception("API returned empty response")
        else:
            raise Exception("API returned invalid response")
