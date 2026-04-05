"""Provider factory for LLM services."""
from typing import Dict, Type, Any, Optional
import importlib
import logging

from .base import BaseProvider
from .ollama import OllamaProvider

# Map of provider names to their respective classes
PROVIDER_CLASSES = {
    "ollama": OllamaProvider,
    "vllm": "llm_mcp.providers.vllm_v1.provider.VLLMv1Provider",
    "vllm_v1": "llm_mcp.providers.vllm_v1.provider.VLLMv1Provider",
    # Add other providers here as they are implemented
}

logger = logging.getLogger(__name__)

class ProviderFactory:
    """Factory class for creating and managing LLM providers."""
    
    @staticmethod
    def create_provider(provider_name: str, config: Dict[str, Any]) -> BaseProvider:
        """Create a new provider instance.
        
        Args:
            provider_name: Name of the provider (e.g., 'ollama', 'vllm')
            config: Configuration dictionary for the provider
            
        Returns:
            An instance of the requested provider
            
        Raises:
            ValueError: If the provider is not supported
            ImportError: If the provider module cannot be imported
        """
        provider_name = provider_name.lower()
        
        # Try to get the provider class from the built-in providers
        provider_class_or_path = PROVIDER_CLASSES.get(provider_name)
        
        # If it's a string path, import the class
        if isinstance(provider_class_or_path, str):
            try:
                module_path, class_name = provider_class_or_path.rsplit('.', 1)
                module = importlib.import_module(module_path)
                provider_class = getattr(module, class_name)
            except (ImportError, AttributeError, ValueError) as e:
                logger.error(f"Failed to import provider {provider_name} from {provider_class_or_path}: {str(e)}")
                raise ImportError(f"Failed to import provider {provider_name}: {str(e)}")
        else:
            provider_class = provider_class_or_path
        
        # If still not found, try to import it dynamically using the provider name
        if provider_class is None:
            try:
                module = importlib.import_module(f"llm_mcp.providers.{provider_name}")
                provider_class = getattr(module, f"{provider_name.capitalize()}Provider", None)
            except (ImportError, AttributeError) as e:
                logger.debug(f"Failed to import provider {provider_name}: {str(e)}")
        
        if provider_class is None or not issubclass(provider_class, BaseProvider):
            raise ValueError(f"Unsupported provider: {provider_name}")
        
        return provider_class(config)
    
    @staticmethod
    def get_available_providers() -> Dict[str, Type[BaseProvider]]:
        """Get a dictionary of all available provider classes.
        
        Returns:
            Dictionary mapping provider names to their classes
        """
        return PROVIDER_CLASSES.copy()
    
    @staticmethod
    def is_provider_available(provider_name: str) -> bool:
        """Check if a provider is available.
        
        Args:
            provider_name: Name of the provider to check
            
        Returns:
            True if the provider is available, False otherwise
        """
        return provider_name.lower() in PROVIDER_CLASSES
    
    @classmethod
    async def create_and_verify_provider(
        cls,
        provider_name: str,
        config: Dict[str, Any]
    ) -> Optional[BaseProvider]:
        """Create a provider and verify it's ready to use.
        
        Args:
            provider_name: Name of the provider to create
            config: Configuration for the provider
            
        Returns:
            The created provider if successful, None otherwise
            
        Raises:
            Exception: If there's an error creating or verifying the provider
        """
        try:
            provider = cls.create_provider(provider_name, config)
            
            # Verify the provider is ready
            if not provider.is_ready:
                logger.warning(f"Provider {provider_name} is not ready")
                return None
                
            # Test listing models to verify connectivity
            try:
                await provider.list_models()
                return provider
            except Exception as e:
                logger.error(f"Failed to list models for provider {provider_name}: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating provider {provider_name}: {str(e)}")
            raise
