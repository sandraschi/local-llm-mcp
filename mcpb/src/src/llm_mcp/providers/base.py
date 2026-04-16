"""Base provider interface for LLM services."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the provider with configuration.

        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config

    @abstractmethod
    async def list_models(self) -> list[dict[str, Any]]:
        """List all available models from the provider.

        Returns:
            List of model information dictionaries, each containing at least:
            - id: str - Unique model identifier
            - name: str - Human-readable model name
            - description: str - Model description
            - capabilities: List[str] - Supported capabilities
        """
        pass

    @abstractmethod
    async def generate(
        self, prompt: str, model: str, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate a response from the model.

        Args:
            prompt: The input prompt
            model: The model to use for generation
            **kwargs: Additional generation parameters

        Yields:
            Chunks of the generated response as strings
        """
        pass

    @abstractmethod
    async def pull_model(self, model_name: str) -> dict[str, Any]:
        """Download a model if it's not already available locally.

        Args:
            model_name: Name of the model to download

        Returns:
            Status information about the download operation
        """
        pass

    @abstractmethod
    async def get_model_info(self, model_name: str) -> dict[str, Any]:
        """Get detailed information about a specific model.

        Args:
            model_name: Name of the model to get info for

        Returns:
            Detailed model information
        """
        pass

    async def generate_text(self, model_id: str, prompt: str, **kwargs) -> str:
        """Generate a full text response (non-streaming).

        Default implementation collects all chunks from the generate method.
        """
        response = []
        async for chunk in self.generate(prompt, model_id, **kwargs):
            response.append(chunk)
        return "".join(response)

    async def chat(
        self, model_id: str, messages: list[dict[str, str]], **kwargs
    ) -> str:
        """Generate a chat response.

        Default implementation uses the latest user message as a prompt.
        Subclasses should override this for proper chat support.
        """
        # Find the last user message
        prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break

        return await self.generate_text(model_id, prompt, **kwargs)

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the provider."""
        pass

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Return whether the provider supports streaming responses."""
        return True

    @property
    def is_ready(self) -> bool:
        """Check if the provider is ready to handle requests.

        Can be overridden by subclasses to perform readiness checks.
        """
        return True
