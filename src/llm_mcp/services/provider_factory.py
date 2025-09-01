"""Factory for creating and managing LLM providers."""
from typing import Dict, Optional, Type, Any

from ..models.base import BaseProvider, ModelProvider
from ..providers.ollama import OllamaProvider
from ..providers.lmstudio import LMStudioProvider
from ..providers.vllm import VLLMProvider
from ..providers.anthropic import AnthropicProvider
from ..providers.huggingface import HuggingFaceProvider

# Map provider types to their implementation classes
PROVIDER_CLASSES: Dict[ModelProvider, Type[BaseProvider]] = {
    ModelProvider.OLLAMA: OllamaProvider,
    ModelProvider.LMSTUDIO: LMStudioProvider,
    ModelProvider.VLLM: VLLMProvider,
    ModelProvider.ANTHROPIC: AnthropicProvider,
    ModelProvider.HUGGINGFACE: HuggingFaceProvider,
    # Add other providers here as they are implemented
    # ModelProvider.OPENAI: OpenAIProvider,
    # ModelProvider.GEMINI: GeminiProvider,
    # ModelProvider.PERPLEXITY: PerplexityProvider,
}

class ProviderFactory:
    """Factory for creating and managing LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the provider factory.
        
        Args:
            config: Configuration dictionary with provider-specific settings
        """
        self.config = config
        self._providers: Dict[ModelProvider, BaseProvider] = {}
    
    def get_provider(self, provider_type: ModelProvider) -> BaseProvider:
        """Get or create a provider instance.
        
        Args:
            provider_type: Type of the provider to get
            
        Returns:
            An instance of the requested provider
            
        Raises:
            ValueError: If the provider type is not supported
        """
        if provider_type not in PROVIDER_CLASSES:
            raise ValueError(f"Unsupported provider type: {provider_type}")
        
        if provider_type not in self._providers:
            # Get provider-specific config with fallback to empty dict
            provider_config = self.config.get(provider_type.value, {})
            
            # Create a new provider instance
            provider_class = PROVIDER_CLASSES[provider_type]
            self._providers[provider_type] = provider_class(provider_config)
        
        return self._providers[provider_type]
    
    async def get_provider_for_model(self, model_id: str) -> Optional[BaseProvider]:
        """Get the appropriate provider for a given model ID.
        
        This method tries to determine the provider based on the model ID format
        or by checking each available provider.
        
        Args:
            model_id: ID of the model
            
        Returns:
            The provider instance if found, None otherwise
        """
        # First, try to determine provider from model ID format
        if model_id.startswith("ollama:"):
            return self.get_provider(ModelProvider.OLLAMA)
        # Add more provider-specific patterns here
        
        # If we couldn't determine the provider from the ID, try each provider
        for provider_type in PROVIDER_CLASSES:
            try:
                provider = self.get_provider(provider_type)
                # Check if the model exists with this provider
                # This is a simple check and might be optimized
                models = await provider.list_models()
                if any(model.id == model_id or model.name == model_id for model in models):
                    return provider
            except Exception:
                # Skip this provider and try the next one
                continue
        
        return None
    
    async def get_all_models(self) -> Dict[ModelProvider, list]:
        """Get all models from all available providers.
        
        Returns:
            A dictionary mapping provider types to lists of their models
        """
        all_models = {}
        
        for provider_type in PROVIDER_CLASSES:
            try:
                provider = self.get_provider(provider_type)
                models = await provider.list_models()
                all_models[provider_type] = models
            except Exception as e:
                print(f"Error getting models from {provider_type}: {str(e)}")
                all_models[provider_type] = []
        
        return all_models
    
    async def close(self):
        """Clean up resources used by providers."""
        for provider in self._providers.values():
            if hasattr(provider, 'close'):
                await provider.close()
        self._providers.clear()
