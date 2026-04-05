"""OpenAI provider implementation."""

import asyncio
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
import json
import time

import aiohttp
from pydantic import BaseModel

from llm_mcp.models.base import BaseProvider, ModelMetadata, ModelProvider, ModelCapability, ModelStatus

logger = logging.getLogger(__name__)

# Try to import openai, but make it optional
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI not installed. Install with: pip install openai")
    OPENAI_AVAILABLE = False

class OpenAIProvider(BaseProvider):
    """
    OpenAI provider for GPT models.
    
    Features:
    - GPT-4, GPT-4o, GPT-3.5 Turbo
    - Streaming and non-streaming generation
    - Function calling support
    - Vision capabilities (GPT-4V)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the OpenAI provider.
        
        Args:
            config: Configuration dictionary for the OpenAI provider
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI is not installed. Please install it with: pip install openai")
            
        from .config import OpenAIConfig
        self.config = OpenAIConfig(**(config or {}))
        
        # Initialize OpenAI client
        self.client = openai.AsyncOpenAI(
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
        return "openai"
    
    @property
    def is_ready(self) -> bool:
        """Check if the provider is ready to handle requests."""
        return self._is_initialized and self.config.api_key is not None
    
    async def initialize(self) -> None:
        """Initialize the OpenAI provider."""
        if self._is_initialized:
            return
            
        logger.info("Initializing OpenAI provider")
        
        try:
            # Test the connection
            if self.config.api_key:
                await self._test_connection()
            
            self._is_initialized = True
            logger.info("OpenAI provider initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize OpenAI provider: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.metrics["last_error"] = error_msg
            raise RuntimeError(error_msg) from e
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        self._is_initialized = False
        logger.info("OpenAI provider cleaned up")
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available OpenAI models.
        
        Returns:
            List of model information dictionaries
        """
        if not self.is_ready:
            await self.initialize()
            
        models = [
            {
                "id": "gpt-4o",
                "name": "GPT-4o",
                "description": "Latest GPT-4 model with vision capabilities",
                "capabilities": ["text-generation", "chat", "vision"],
                "max_tokens": 128000,
                "context_length": 128000,
                "provider": "openai"
            },
            {
                "id": "gpt-4o-mini",
                "name": "GPT-4o Mini",
                "description": "Faster, cheaper version of GPT-4o",
                "capabilities": ["text-generation", "chat", "vision"],
                "max_tokens": 128000,
                "context_length": 128000,
                "provider": "openai"
            },
            {
                "id": "gpt-4-turbo",
                "name": "GPT-4 Turbo",
                "description": "Previous generation GPT-4 with large context",
                "capabilities": ["text-generation", "chat", "vision"],
                "max_tokens": 128000,
                "context_length": 128000,
                "provider": "openai"
            },
            {
                "id": "gpt-4",
                "name": "GPT-4",
                "description": "Original GPT-4 model",
                "capabilities": ["text-generation", "chat"],
                "max_tokens": 8192,
                "context_length": 8192,
                "provider": "openai"
            },
            {
                "id": "gpt-3.5-turbo",
                "name": "GPT-3.5 Turbo",
                "description": "Fast and efficient GPT-3.5 model",
                "capabilities": ["text-generation", "chat"],
                "max_tokens": 16384,
                "context_length": 16384,
                "provider": "openai"
            }
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
                "frequency_penalty": kwargs.get("frequency_penalty", self.config.frequency_penalty),
                "presence_penalty": kwargs.get("presence_penalty", self.config.presence_penalty),
                "stop": kwargs.get("stop", self.config.stop),
                "user": kwargs.get("user", self.config.user),
                "stream": True
            }
            
            # Generate text using OpenAI streaming
            stream = await self.client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                **generation_params
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            
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
                "frequency_penalty": kwargs.get("frequency_penalty", self.config.frequency_penalty),
                "presence_penalty": kwargs.get("presence_penalty", self.config.presence_penalty),
                "stop": kwargs.get("stop", self.config.stop),
                "user": kwargs.get("user", self.config.user)
            }
            
            # Generate response
            response = await self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                **generation_params
            )
            
            # Update metrics
            duration = time.time() - start_time
            self.metrics["successful_requests"] += 1
            self.metrics["total_tokens_generated"] += len(response.choices[0].message.content.split())
            self.metrics["total_time_seconds"] += duration
            
            return response.choices[0].message.content
            
        except Exception as e:
            error_msg = f"Error in chat completion: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.metrics["failed_requests"] += 1
            self.metrics["last_error"] = error_msg
            raise RuntimeError(error_msg) from e
    
    async def pull_model(self, model_name: str) -> Dict[str, Any]:
        """Pull a model (not applicable for OpenAI API).
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information
        """
        logger.info(f"OpenAI models are API-based, no pulling needed for {model_name}")
        
        # Return model info from available models
        models = await self.list_models()
        for model in models:
            if model["id"] == model_name:
                return model
        
        raise ValueError(f"Model {model_name} not found in available OpenAI models")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get provider metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = self.metrics.copy()
        metrics.update({
            "provider": "openai",
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
            "provider": "openai",
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
        
        raise ValueError(f"Model {model_name} not found in available OpenAI models")
    
    @property
    def supports_streaming(self) -> bool:
        """Return whether the provider supports streaming responses."""
        return True
    
    async def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get details about a specific model."""
        models = await self.list_models()
        for model in models:
            if model["id"] == model_id:
                return ModelMetadata(
                    id=model["id"],
                    name=model["name"],
                    provider=ModelProvider.OPENAI,
                    capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT],
                    parameters={"max_tokens": model.get("max_tokens", 4096)}
                )
        return None
    
    async def load_model(self, model_id: str, **kwargs) -> ModelMetadata:
        """Load a model into memory."""
        model = await self.get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        model.status = ModelStatus.LOADED
        return model
    
    async def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory."""
        model = await self.get_model(model_id)
        if model:
            model.status = ModelStatus.UNLOADED
            return True
        return False
    
    async def generate_text(self, model_id: str, prompt: str, **kwargs) -> str:
        """Generate text using the specified model."""
        result = ""
        async for chunk in self.generate(prompt, model_id, **kwargs):
            result += chunk
        return result
    
    async def chat(self, model_id: str, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a chat completion using the specified model."""
        response = await self.chat_completion(model_id=model_id, messages=messages, **kwargs)
        return response["content"]
    
    async def generate_embeddings(self, model_id: str, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for the given texts."""
        # OpenAI supports embeddings
        try:
            response = await self.client.embeddings.create(
                model=model_id,
                input=texts,
                **kwargs
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            raise Exception(f"Failed to generate embeddings: {str(e)}") from e
