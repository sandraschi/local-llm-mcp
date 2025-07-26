"""Model management service for LLM MCP Server."""
from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime

from ..models.base import ModelMetadata, ModelProvider, ModelStatus
from .provider_factory import ProviderFactory

class ModelManager:
    """Manages LLM models across different providers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model manager.
        
        Args:
            config: Configuration dictionary with provider-specific settings
        """
        self.provider_factory = ProviderFactory(config)
        self._models_cache: Dict[str, ModelMetadata] = {}
        self._models_lock = asyncio.Lock()
        self._initialized = False
    
    async def initialize(self):
        """Initialize the model manager and load available models."""
        if self._initialized:
            return
        
        # Load all models from all providers
        await self.refresh_models()
        self._initialized = True
    
    async def refresh_models(self):
        """Refresh the list of available models from all providers."""
        async with self._models_lock:
            all_models = await self.provider_factory.get_all_models()
            
            # Clear the cache
            self._models_cache.clear()
            
            # Update the cache with models from all providers
            for provider_type, models in all_models.items():
                for model in models:
                    self._models_cache[model.id] = model
    
    async def list_models(self, provider: Optional[str] = None) -> List[ModelMetadata]:
        """List all available models, optionally filtered by provider.
        
        Args:
            provider: Optional provider name to filter by
            
        Returns:
            List of model metadata objects
        """
        if not self._initialized:
            await self.initialize()
        
        if provider:
            # Try to convert provider string to enum
            try:
                provider_enum = ModelProvider(provider.lower())
                return [
                    model for model in self._models_cache.values() 
                    if model.provider == provider_enum
                ]
            except ValueError:
                # If provider is not a valid enum value, return empty list
                return []
        
        return list(self._models_cache.values())
    
    async def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get details about a specific model.
        
        Args:
            model_id: ID of the model to get
            
        Returns:
            Model metadata if found, None otherwise
        """
        if not self._initialized:
            await self.initialize()
        
        # Check cache first
        if model_id in self._models_cache:
            return self._models_cache[model_id]
        
        # If not in cache, try to find the model by querying providers
        provider = await self.provider_factory.get_provider_for_model(model_id)
        if provider:
            model = await provider.get_model(model_id)
            if model:
                self._models_cache[model_id] = model
                return model
        
        return None
    
    async def load_model(self, model_id: str, **kwargs) -> ModelMetadata:
        """Load a model into memory.
        
        Args:
            model_id: ID of the model to load
            **kwargs: Additional parameters for the model
            
        Returns:
            Updated model metadata
            
        Raises:
            ValueError: If the model is not found or cannot be loaded
        """
        if not self._initialized:
            await self.initialize()
        
        # Get the provider for this model
        provider = await self.provider_factory.get_provider_for_model(model_id)
        if not provider:
            raise ValueError(f"Could not find provider for model: {model_id}")
        
        try:
            # Update model status to loading
            model = await self.get_model(model_id)
            if model:
                model.status = ModelStatus.LOADING
                self._models_cache[model_id] = model
            
            # Load the model
            model = await provider.load_model(model_id, **kwargs)
            
            # Update the cache
            if model:
                self._models_cache[model_id] = model
                return model
            
            raise ValueError(f"Failed to load model: {model_id}")
        except Exception as e:
            # Update model status to error
            if model_id in self._models_cache:
                self._models_cache[model_id].status = ModelStatus.ERROR
            raise ValueError(f"Error loading model {model_id}: {str(e)}") from e
    
    async def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory.
        
        Args:
            model_id: ID of the model to unload
            
        Returns:
            True if the model was unloaded, False otherwise
        """
        if not self._initialized:
            await self.initialize()
        
        # Get the provider for this model
        provider = await self.provider_factory.get_provider_for_model(model_id)
        if not provider:
            return False
        
        try:
            # Unload the model
            success = await provider.unload_model(model_id)
            
            # Update model status if unload was successful
            if success and model_id in self._models_cache:
                self._models_cache[model_id].status = ModelStatus.UNLOADED
            
            return success
        except Exception as e:
            print(f"Error unloading model {model_id}: {str(e)}")
            return False
    
    async def generate_text(
        self,
        model_id: str,
        prompt: str,
        **kwargs
    ) -> str:
        """Generate text using the specified model.
        
        Args:
            model_id: ID of the model to use
            prompt: The prompt to generate text from
            **kwargs: Additional parameters for text generation
            
        Returns:
            Generated text
            
        Raises:
            ValueError: If the model is not found or generation fails
        """
        if not self._initialized:
            await self.initialize()
        
        # Get the provider for this model
        provider = await self.provider_factory.get_provider_for_model(model_id)
        if not provider:
            raise ValueError(f"Could not find provider for model: {model_id}")
        
        try:
            # Ensure the model is loaded
            model = await self.get_model(model_id)
            if model and model.status != ModelStatus.LOADED:
                await self.load_model(model_id)
            
            # Generate text
            return await provider.generate_text(model_id, prompt, **kwargs)
        except Exception as e:
            raise ValueError(f"Error generating text with model {model_id}: {str(e)}") from e
    
    async def chat(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Generate a chat completion using the specified model.
        
        Args:
            model_id: ID of the model to use
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional parameters for chat completion
            
        Returns:
            Generated chat response
            
        Raises:
            ValueError: If the model is not found or chat fails
        """
        if not self._initialized:
            await self.initialize()
        
        # Get the provider for this model
        provider = await self.provider_factory.get_provider_for_model(model_id)
        if not provider:
            raise ValueError(f"Could not find provider for model: {model_id}")
        
        try:
            # Ensure the model is loaded
            model = await self.get_model(model_id)
            if model and model.status != ModelStatus.LOADED:
                await self.load_model(model_id)
            
            # Generate chat completion
            return await provider.chat(model_id, messages, **kwargs)
        except Exception as e:
            raise ValueError(f"Error in chat with model {model_id}: {str(e)}") from e
    
    async def close(self):
        """Clean up resources used by the model manager."""
        await self.provider_factory.close()
        self._models_cache.clear()
        self._initialized = False
