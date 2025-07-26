"""Service for managing LLM models and providers."""
import logging
from typing import Dict, Any, Optional, AsyncGenerator, List

from ..providers import ProviderFactory
from ..config import get_settings, Settings

logger = logging.getLogger(__name__)

class ModelService:
    """Service for managing LLM models and providers."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the model service.
        
        Args:
            settings: Application settings. If not provided, will use default settings.
        """
        self.settings = settings or get_settings()
        self.providers: Dict[str, Any] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the model service and all providers."""
        if self._initialized:
            return
            
        logger.info("Initializing ModelService...")
        
        # Initialize all configured providers
        for provider_name, config in self.settings.providers.items():
            if not config.get("enabled", True):
                logger.info(f"Skipping disabled provider: {provider_name}")
                continue
                
            try:
                provider = await ProviderFactory.create_and_verify_provider(
                    provider_name,
                    config
                )
                
                if provider:
                    self.providers[provider_name] = provider
                    logger.info(f"Initialized provider: {provider_name}")
                else:
                    logger.warning(f"Failed to initialize provider: {provider_name}")
                    
            except Exception as e:
                logger.error(f"Error initializing provider {provider_name}: {str(e)}", exc_info=True)
        
        self._initialized = True
        logger.info("ModelService initialization complete")
    
    async def list_models(self, provider_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all available models from all or a specific provider.
        
        Args:
            provider_name: Optional name of the provider to list models from.
                          If None, lists models from all providers.
                          
        Returns:
            List of model information dictionaries.
        """
        if not self._initialized:
            await self.initialize()
            
        models = []
        
        # If a specific provider is requested
        if provider_name:
            provider = self.providers.get(provider_name.lower())
            if not provider:
                raise ValueError(f"Provider not found or not initialized: {provider_name}")
                
            provider_models = await provider.list_models()
            for model in provider_models:
                model["provider"] = provider_name
                models.append(model)
        else:
            # List models from all providers
            for provider_name, provider in self.providers.items():
                try:
                    provider_models = await provider.list_models()
                    for model in provider_models:
                        model["provider"] = provider_name
                        models.append(model)
                except Exception as e:
                    logger.error(f"Error listing models from provider {provider_name}: {str(e)}", exc_info=True)
        
        return models
    
    async def generate(
        self,
        prompt: str,
        model: str,
        provider_name: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate text using the specified model and provider.
        
        Args:
            prompt: The input prompt
            model: The model to use for generation
            provider_name: Optional name of the provider to use. If None, uses the default provider.
            **kwargs: Additional generation parameters
            
        Yields:
            Chunks of the generated text
            
        Raises:
            ValueError: If the provider or model is not found
        """
        if not self._initialized:
            await self.initialize()
        
        # If no provider specified, use the default one
        if not provider_name:
            provider_name = self.settings.default_provider
        
        provider = self.providers.get(provider_name.lower())
        if not provider:
            raise ValueError(f"Provider not found or not initialized: {provider_name}")
        
        # Check if the model exists with this provider
        try:
            models = await self.list_models(provider_name)
            model_exists = any(m["name"] == model for m in models)
            
            if not model_exists:
                # Try to pull the model if it doesn't exist
                if hasattr(provider, "pull_model"):
                    logger.info(f"Model {model} not found. Attempting to pull...")
                    try:
                        await provider.pull_model(model)
                        logger.info(f"Successfully pulled model: {model}")
                    except Exception as e:
                        logger.error(f"Failed to pull model {model}: {str(e)}")
                        raise ValueError(f"Model {model} not found and could not be pulled")
                else:
                    raise ValueError(f"Model {model} not found with provider {provider_name}")
            
            # Generate the response
            async for chunk in provider.generate(prompt, model, **kwargs):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error during generation with {provider_name}/{model}: {str(e)}", exc_info=True)
            raise
    
    async def get_model_info(self, model_name: str, provider_name: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model
            provider_name: Optional name of the provider. If None, searches all providers.
            
        Returns:
            Detailed model information
            
        Raises:
            ValueError: If the model is not found
        """
        if not self._initialized:
            await self.initialize()
        
        # If provider is specified, only check that provider
        if provider_name:
            provider = self.providers.get(provider_name.lower())
            if not provider:
                raise ValueError(f"Provider not found: {provider_name}")
                
            try:
                model_info = await provider.get_model_info(model_name)
                model_info["provider"] = provider_name
                return model_info
            except Exception as e:
                raise ValueError(f"Error getting model info: {str(e)}")
        
        # Otherwise, search all providers
        for provider_name, provider in self.providers.items():
            try:
                model_info = await provider.get_model_info(model_name)
                model_info["provider"] = provider_name
                return model_info
            except (ValueError, KeyError):
                continue
            except Exception as e:
                logger.warning(f"Error getting model info from {provider_name}: {str(e)}")
        
        raise ValueError(f"Model not found: {model_name}")
    
    async def close(self) -> None:
        """Clean up resources."""
        for provider in self.providers.values():
            if hasattr(provider, 'close') and callable(provider.close):
                await provider.close()
        
        self.providers.clear()
        self._initialized = False


# Create a singleton instance of the model service
model_service = ModelService()
