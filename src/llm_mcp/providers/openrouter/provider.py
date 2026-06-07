"""OpenRouter provider implementation."""

import logging
from collections.abc import AsyncGenerator
from typing import Any

from llm_mcp.models.base import BaseProvider

logger = logging.getLogger(__name__)

# Re-use openai library for OpenRouter as it is OpenAI-compatible
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI not installed, required for OpenRouter. Install with: pip install openai")
    OPENAI_AVAILABLE = False


class OpenRouterProvider(BaseProvider):
    """
    OpenRouter provider for unified access to multiple LLMs.

    Features:
    - Access to Gemma 2, Llama 3, Claude, GPT-4, etc.
    - OpenAI-compatible API
    - Vision support for capable models
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the OpenRouter provider.

        Args:
            config: Configuration dictionary for the OpenRouter provider
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI is not installed. Please install it with: pip install openai")

        from .config import OpenRouterConfig

        self.config = OpenRouterConfig(**(config or {}))

        # Initialize client with OpenRouter base URL and headers
        self.client = openai.AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            default_headers={
                "HTTP-Referer": self.config.site_url or "https://github.com/google-deepmind/local-llm-mcp",
                "X-Title": self.config.site_name or "Local LLM MCP",
            },
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )

        self._is_ready = self.config.api_key is not None

    @property
    def name(self) -> str:
        return "openrouter"

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    async def initialize(self) -> None:
        """Initialize the OpenRouter provider."""
        if self._is_ready:
            return

        if not self.config.api_key:
            logger.warning("OpenRouter API key not configured")
            return

        self._is_ready = True
        logger.info("OpenRouter provider initialized")

    async def list_models(self) -> list[dict[str, Any]]:
        """List available models via OpenRouter."""
        # For simplicity, we return a curated list, though OpenRouter has an endpoint
        # In a production scenario, we'd fetch https://openrouter.ai/api/v1/models
        return [
            {
                "id": "google/gemma-2-9b-it",
                "name": "Gemma 2 9B IT",
                "description": "Google's latest efficient model",
                "capabilities": ["text-generation", "chat"],
                "provider": "openrouter",
            },
            {
                "id": "google/gemma-2-27b-it",
                "name": "Gemma 2 27B IT",
                "description": "High performance Gemma 2 model",
                "capabilities": ["text-generation", "chat"],
                "provider": "openrouter",
            },
            {
                "id": "meta-llama/llama-3.1-8b-instruct",
                "name": "Llama 3.1 8B",
                "description": "Latest Llama 3.1 8B",
                "capabilities": ["text-generation", "chat"],
                "provider": "openrouter",
            },
            {
                "id": "anthropic/claude-3.5-sonnet",
                "name": "Claude 3.5 Sonnet",
                "description": "Anthropic's latest high-IQ model",
                "capabilities": ["text-generation", "chat", "vision"],
                "provider": "openrouter",
            },
        ]

    async def generate(self, prompt: str, model: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate text from OpenRouter."""
        if not self.is_ready:
            await self.initialize()

        model_id = model or self.config.default_model

        try:
            stream = await self.client.chat.completions.create(
                model=model_id, messages=[{"role": "user", "content": prompt}], stream=True, **kwargs
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenRouter generation error: {e!s}")
            raise

    async def chat_completion(self, messages: list[dict[str, str]], model: str | None = None, **kwargs) -> str:
        """Generate chat completion from OpenRouter."""
        model_id = model or self.config.default_model
        try:
            response = await self.client.chat.completions.create(model=model_id, messages=messages, **kwargs)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenRouter chat error: {e!s}")
            raise

    async def pull_model(self, model_name: str) -> dict[str, Any]:
        """Cloud API, no pulling required."""
        return {"status": "available", "id": model_name}

    async def health_check(self) -> dict[str, Any]:
        """Check OpenRouter connectivity."""
        return {
            "status": "healthy" if self.is_ready else "unconfigured",
            "provider": "openrouter",
            "api_key_set": self.config.api_key is not None,
        }
