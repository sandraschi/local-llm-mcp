"""Base provider interface for LLM services."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncGenerator, Optional


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the provider with configuration.
        
        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config

    @abstractmethod
    async def list_models(self) -> List[Dict[str, Any]]:
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
        self,
        prompt: str,
        model: str,
        **kwargs
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
    async def pull_model(self, model_name: str) -> Dict[str, Any]:
        """Download a model if it's not already available locally.
        
        Args:
            model_name: Name of the model to download
            
        Returns:
            Status information about the download operation
        """
        pass

    @abstractmethod
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model to get info for
            
        Returns:
            Detailed model information
        """
        pass

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
