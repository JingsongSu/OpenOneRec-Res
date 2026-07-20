"""
Gemini API Client Implementation
Based on Google Vertex AI's Gemini model
"""
import os
from typing import Optional
from vertexai.generative_models import GenerativeModel
import vertexai
from .base import BaseLLMClient


class GeminiClient(BaseLLMClient):
    """
    Gemini API Client

    Example:
        >>> client = GeminiClient(
        ...     project="your-project",
        ...     location="us-central1",
        ...     model_name="gemini-2.5-pro",
        ...     credentials_path="path/to/credentials.json"
        ... )
        >>> response = client.generate("Tell me a joke")
    """

    def _setup(self):
        """Initialize Gemini client"""
        self.project = self.config.get("project")
        self.location = self.config.get("location")
        self.model_name = self.config.get("model_name", "gemini-2.5-pro")
        credentials_path = self.config.get("credentials_path")
        self.default_max_tokens = self.config.get("max_new_tokens")
        self.default_temperature = self.config.get("temperature")

        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

        if not self.project or not self.location:
            raise ValueError("project and location are required parameters")

        vertexai.init(project=self.project, location=self.location)
        self.model = GenerativeModel(self.model_name)

    def _call_api(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Call Gemini API to generate text

        Args:
            prompt: Input prompt
            temperature: Temperature parameter (0.0-1.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Other Gemini-specific parameters

        Returns:
            str: Generated text content

        Raises:
            Exception: Raised when API call fails
        """
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        generation_config = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens

        if generation_config:
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
        else:
            response = self.model.generate_content(prompt)

        if response and response.text:
            return response.text
        else:
            raise Exception("API returned empty response")
