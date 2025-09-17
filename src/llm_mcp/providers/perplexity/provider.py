"""Perplexity provider implementation."""

import asyncio
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
import json
import time

import aiohttp
from pydantic import BaseModel

from llm_mcp.models.base import BaseProvider, ModelMetadata, ModelProvider, ModelCapability, ModelStatus

logger = logging.getLogger(__name__)

class PerplexityProvider(BaseProvider):
    """
    Perplexity provider for Perplexity AI models.
    
    Features:
    - Sonar models with web search capabilities
    - Real-time information access
    - Streaming and non-streaming generation
    - Online and offline modes
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Perplexity provider.
        
        Args:
            config: Configuration dictionary for the Perplexity provider
        """
        from .config import PerplexityConfig
        self.config = PerplexityConfig(**(config or {}))
        
        # Don't create session in __init__ to avoid event loop issues
        self.session = None
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
        return "perplexity"
    
    @property
    def is_ready(self) -> bool:
        """Check if the provider is ready to handle requests."""
        return self._is_initialized and self.config.api_key is not None
    
    async def initialize(self) -> None:
        """Initialize the Perplexity provider."""
        if self._is_initialized:
            return
            
        logger.info("Initializing Perplexity provider")
        
        # Create HTTP session if not already created
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                headers={
                    "Authorization": f"Bearer {self.config.api_key}" if self.config.api_key else None,
                    "Content-Type": "application/json"
                }
            )
        
        try:
            # Test the connection
            if self.config.api_key:
                await self._test_connection()
            
            self._is_initialized = True
            logger.info("Perplexity provider initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize Perplexity provider: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.metrics["last_error"] = error_msg
            raise RuntimeError(error_msg) from e
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if hasattr(self, 'session'):
            await self.session.close()
        self._is_initialized = False
        logger.info("Perplexity provider cleaned up")
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available Perplexity models.
        
        Returns:
            List of model information dictionaries
        """
        if not self.is_ready:
            await self.initialize()
            
        models = [
            {
                "id": "llama-3.1-sonar-large-128k-online",
                "name": "Llama 3.1 Sonar Large (Online)",
                "description": "Most capable Perplexity model with web search",
                "capabilities": ["text-generation", "chat", "web-search"],
                "max_tokens": 4096,
                "context_length": 128000,
                "provider": "perplexity",
                "online": True
            },
            {
                "id": "llama-3.1-sonar-small-128k-online",
                "name": "Llama 3.1 Sonar Small (Online)",
                "description": "Fast Perplexity model with web search",
                "capabilities": ["text-generation", "chat", "web-search"],
                "max_tokens": 4096,
                "context_length": 128000,
                "provider": "perplexity",
                "online": True
            },
            {
                "id": "llama-3.1-sonar-large-128k-chat",
                "name": "Llama 3.1 Sonar Large (Offline)",
                "description": "Most capable Perplexity model without web search",
                "capabilities": ["text-generation", "chat"],
                "max_tokens": 4096,
                "context_length": 128000,
                "provider": "perplexity",
                "online": False
            },
            {
                "id": "llama-3.1-sonar-small-128k-chat",
                "name": "Llama 3.1 Sonar Small (Offline)",
                "description": "Fast Perplexity model without web search",
                "capabilities": ["text-generation", "chat"],
                "max_tokens": 4096,
                "context_length": 128000,
                "provider": "perplexity",
                "online": False
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
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "top_k": kwargs.get("top_k", self.config.top_k),
                "stop": kwargs.get("stop", self.config.stop),
                "stream": True
            }
            
            # Generate text using Perplexity streaming
            async with self.session.post(
                f"{self.config.base_url}/chat/completions",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Perplexity API error: {response.status} - {error_text}")
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    yield delta['content']
                        except json.JSONDecodeError:
                            continue
            
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
            payload = {
                "model": model_id,
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "top_k": kwargs.get("top_k", self.config.top_k),
                "stop": kwargs.get("stop", self.config.stop),
                "stream": False
            }
            
            # Generate response
            async with self.session.post(
                f"{self.config.base_url}/chat/completions",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Perplexity API error: {response.status} - {error_text}")
                
                data = await response.json()
                response_text = data["choices"][0]["message"]["content"]
            
            # Update metrics
            duration = time.time() - start_time
            self.metrics["successful_requests"] += 1
            self.metrics["total_tokens_generated"] += len(response_text.split())
            self.metrics["total_time_seconds"] += duration
            
            return response_text
            
        except Exception as e:
            error_msg = f"Error in chat completion: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.metrics["failed_requests"] += 1
            self.metrics["last_error"] = error_msg
            raise RuntimeError(error_msg) from e
    
    async def pull_model(self, model_name: str) -> Dict[str, Any]:
        """Pull a model (not applicable for Perplexity API).
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information
        """
        logger.info(f"Perplexity models are API-based, no pulling needed for {model_name}")
        
        # Return model info from available models
        models = await self.list_models()
        for model in models:
            if model["id"] == model_name:
                return model
        
        raise ValueError(f"Model {model_name} not found in available Perplexity models")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get provider metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = self.metrics.copy()
        metrics.update({
            "provider": "perplexity",
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
            "provider": "perplexity",
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
        
        raise ValueError(f"Model {model_name} not found in available Perplexity models")
    
    @property
    def supports_streaming(self) -> bool:
        """Return whether the provider supports streaming responses."""
        return True
    

    async def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get details about a specific model."""
        models = await self.list_models()
        for model in models:
            if isinstance(model, dict) and model.get("id") == model_id:
                return ModelMetadata(
                    id=model["id"],
                    name=model["name"],
                    provider=ModelProvider.GEMINI if "gemini" in file_path else ModelProvider.PERPLEXITY,
                    capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT],
                    parameters={"max_tokens": model.get("max_tokens", 4096)}
                )
            elif hasattr(model, 'id') and model.id == model_id:
                return model
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
        return response.get("content", str(response))
    
    async def generate_embeddings(self, model_id: str, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for the given texts."""
        raise NotImplementedError("Embeddings not supported by this provider")

    async def _test_connection(self) -> None:
        """Test the connection to Perplexity API."""
        try:
            # Make a simple request to test the connection
            payload = {
                "model": "llama-3.1-sonar-small-128k-chat",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1
            }
            
            async with self.session.post(
                f"{self.config.base_url}/chat/completions",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API test failed: {response.status} - {error_text}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Perplexity API: {str(e)}") from e
