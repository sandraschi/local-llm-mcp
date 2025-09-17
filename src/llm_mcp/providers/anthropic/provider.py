"""Anthropic provider implementation."""

import asyncio
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
import json
import time

import aiohttp
from pydantic import BaseModel

from llm_mcp.models.base import BaseProvider, ModelMetadata, ModelProvider, ModelCapability, ModelStatus

logger = logging.getLogger(__name__)

# Try to import anthropic, but make it optional
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    logger.warning("Anthropic not installed. Install with: pip install anthropic")
    ANTHROPIC_AVAILABLE = False

class AnthropicProvider(BaseProvider):
    """
    Anthropic provider for Claude models.
    
    Features:
    - Claude 3 Sonnet, Opus, and Haiku
    - Streaming and non-streaming generation
    - Tool calling support
    - Vision capabilities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Anthropic provider.
        
        Args:
            config: Configuration dictionary for the Anthropic provider
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic is not installed. Please install it with: pip install anthropic")
            
        from .config import AnthropicConfig
        self.config = AnthropicConfig(**(config or {}))
        
        # Initialize Anthropic client
        self.client = anthropic.Anthropic(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries
        )
        
        self._is_initialized = False
        
        # Initialize metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens_generated": 0,
            "total_time_seconds": 0.0,
            "last_error": None
        }
    
    @property
    def name(self) -> str:
        return "anthropic"
    
    @property
    def is_ready(self) -> bool:
        """Check if the provider is ready to handle requests."""
        return self._is_initialized and self.config.api_key is not None
    
    async def initialize(self) -> None:
        """Initialize the Anthropic provider."""
        if self._is_initialized:
            return
            
        logger.info("Initializing Anthropic provider")
        
        try:
            # Test the connection
            if self.config.api_key:
                # Make a simple request to test the connection
                await self._test_connection()
            
            self._is_initialized = True
            logger.info("Anthropic provider initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize Anthropic provider: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.metrics["last_error"] = error_msg
            raise RuntimeError(error_msg) from e
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        self._is_initialized = False
        logger.info("Anthropic provider cleaned up")
    
    async def list_models(self) -> List[ModelMetadata]:
        """List available Anthropic models.
        
        Returns:
            List of model metadata objects
        """
        if not self.is_ready:
            await self.initialize()
            
        models = [
            ModelMetadata(
                id="claude-3-opus-20240229",
                name="Claude 3 Opus",
                provider=ModelProvider.ANTHROPIC,
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT],
                parameters={"max_tokens": 200000, "context_length": 200000}
            ),
            ModelMetadata(
                id="claude-3-sonnet-20240229",
                name="Claude 3 Sonnet",
                provider=ModelProvider.ANTHROPIC,
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT],
                parameters={"max_tokens": 200000, "context_length": 200000}
            ),
            ModelMetadata(
                id="claude-3-haiku-20240307",
                name="Claude 3 Haiku",
                provider=ModelProvider.ANTHROPIC,
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT],
                parameters={"max_tokens": 200000, "context_length": 200000}
            ),
            ModelMetadata(
                id="claude-3-5-sonnet-20241022",
                name="Claude 3.5 Sonnet",
                provider=ModelProvider.ANTHROPIC,
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT],
                parameters={"max_tokens": 200000, "context_length": 200000}
            )
        ]
        
        return models
    
    async def generate(
        self,
        prompt: str,
        model: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate text from the model.
        
        Args:
            prompt: The input prompt
            model: Model to use (defaults to configured model)
            **kwargs: Additional generation parameters
            
        Yields:
            Chunks of generated text
        """
        if not self.is_ready:
            await self.initialize()
            
        model_id = model or self.config.default_model
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        try:
            # Prepare generation parameters
            generation_params = {
                "model": model_id,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "top_k": kwargs.get("top_k", self.config.top_k),
                "stop_sequences": kwargs.get("stop_sequences", self.config.stop_sequences),
                "metadata": kwargs.get("metadata", self.config.metadata),
                "stream": True
            }
            
            # Generate text using Anthropic streaming
            async with self.client.messages.stream(
                model=model_id,
                max_tokens=generation_params["max_tokens"],
                temperature=generation_params["temperature"],
                top_p=generation_params["top_p"],
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                async for text in stream.text_stream:
                    yield text
            
            # Update metrics
            duration = time.time() - start_time
            self.metrics["successful_requests"] += 1
            self.metrics["total_time_seconds"] += duration
            
        except Exception as e:
            error_msg = f"Error in text generation: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.metrics["failed_requests"] += 1
            self.metrics["last_error"] = error_msg
            raise RuntimeError(error_msg) from e
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate chat completion.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use (defaults to configured model)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        if not self.is_ready:
            await self.initialize()
            
        model_id = model or self.config.default_model
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        try:
            # Prepare generation parameters
            generation_params = {
                "model": model_id,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "top_k": kwargs.get("top_k", self.config.top_k),
                "stop_sequences": kwargs.get("stop_sequences", self.config.stop_sequences),
                "metadata": kwargs.get("metadata", self.config.metadata)
            }
            
            # Generate response
            response = await self.client.messages.create(
                model=model_id,
                max_tokens=generation_params["max_tokens"],
                temperature=generation_params["temperature"],
                top_p=generation_params["top_p"],
                messages=messages
            )
            
            # Update metrics
            duration = time.time() - start_time
            self.metrics["successful_requests"] += 1
            self.metrics["total_tokens_generated"] += len(response.content[0].text.split())
            self.metrics["total_time_seconds"] += duration
            
            return response.content[0].text
            
        except Exception as e:
            error_msg = f"Error in chat completion: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.metrics["failed_requests"] += 1
            self.metrics["last_error"] = error_msg
            raise RuntimeError(error_msg) from e
    
    async def pull_model(self, model_name: str) -> Dict[str, Any]:
        """Pull a model (not applicable for Anthropic API).
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information
        """
        logger.info(f"Anthropic models are API-based, no pulling needed for {model_name}")
        
        # Return model info from available models
        models = await self.list_models()
        for model in models:
            if model["id"] == model_name:
                return model
        
        raise ValueError(f"Model {model_name} not found in available Anthropic models")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get provider metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = self.metrics.copy()
        metrics.update({
            "provider": "anthropic",
            "api_key_configured": self.config.api_key is not None,
            "base_url": self.config.base_url,
            "default_model": self.config.default_model
        })
        
        return metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the provider.
        
        Returns:
            Health check results
        """
        status = {
            "status": "healthy" if self.is_ready else "unhealthy",
            "provider": "anthropic",
            "api_key_configured": self.config.api_key is not None,
            "last_error": self.metrics.get("last_error"),
            "total_requests": self.metrics["total_requests"],
            "successful_requests": self.metrics["successful_requests"],
            "failed_requests": self.metrics["failed_requests"],
        }
        
        # Test API connection if possible
        if self.config.api_key:
            try:
                await self._test_connection()
                status["api_connection"] = "healthy"
            except Exception as e:
                status["api_connection"] = "unhealthy"
                status["api_error"] = str(e)
        else:
            status["api_connection"] = "no_api_key"
            
        return status
    
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model to get info for
            
        Returns:
            Detailed model information
        """
        models = await self.list_models()
        for model in models:
            if model["id"] == model_name:
                return model
        
        raise ValueError(f"Model {model_name} not found in available Anthropic models")
    
    @property
    def supports_streaming(self) -> bool:
        """Return whether the provider supports streaming responses."""
        return True
    
    async def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get details about a specific model.
        
        Args:
            model_id: ID of the model to get
            
        Returns:
            Model metadata if found, None otherwise
        """
        models = await self.list_models()
        for model in models:
            if model.id == model_id:
                return model
        return None
    
    async def load_model(self, model_id: str, **kwargs) -> ModelMetadata:
        """Load a model into memory.
        
        Args:
            model_id: ID of the model to load
            **kwargs: Additional loading parameters
            
        Returns:
            Model metadata for the loaded model
        """
        model = await self.get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Anthropic models are always "loaded" (API-based)
        model.status = ModelStatus.LOADED
        return model
    
    async def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory.
        
        Args:
            model_id: ID of the model to unload
            
        Returns:
            True if successful, False otherwise
        """
        # Anthropic models are API-based, so "unloading" is just a status change
        model = await self.get_model(model_id)
        if model:
            model.status = ModelStatus.UNLOADED
            return True
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
            prompt: Text prompt for generation
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Use the existing generate method but collect all chunks
        result = ""
        async for chunk in self.generate(prompt, model_id, **kwargs):
            result += chunk
        return result
    
    async def chat(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Generate a chat completion using the specified model.
        
        Args:
            model_id: ID of the model to use
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        # Convert messages to Anthropic format
        anthropic_messages = []
        system_message = None
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Use the existing chat_completion method
        response = await self.chat_completion(
            model_id=model_id,
            messages=anthropic_messages,
            system=system_message,
            **kwargs
        )
        
        return response["content"]
    
    async def generate_embeddings(
        self,
        model_id: str,
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings for the given texts.
        
        Args:
            model_id: ID of the model to use
            texts: List of texts to embed
            **kwargs: Additional parameters
            
        Returns:
            List of embedding vectors
        """
        # Anthropic doesn't currently support embeddings
        raise NotImplementedError("Anthropic does not support embeddings")
