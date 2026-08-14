"""Perplexity provider implementation."""

import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

import aiohttp

from llm_mcp.models.base import BaseProvider, ModelCapability, ModelMetadata, ModelProvider, ModelStatus

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

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the Perplexity provider.

        Args:
            config: Configuration dictionary for the Perplexity provider
        """
        from .config import PerplexityConfig

        self.config = PerplexityConfig(**(config or {}))

        # Don't create session in __init__ to avoid event loop issues
        self.session: Any = None
        self._is_initialized = False

        # Initialize metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens_generated": 0,
            "total_time_seconds": 0.0,
            "last_error": None,
        }

    @property
    def name(self) -> str:
        return "perplexity"

    @property
    def is_ready(self) -> bool:
        """Check if the provider is ready to handle requests."""
        return self._is_initialized and self.config.api_key is not None  # ty: ignore[unresolved-attribute]

    async def initialize(self) -> None:
        """Initialize the Perplexity provider."""
        if self._is_initialized:
            return

        logger.info("Initializing Perplexity provider")

        # Create HTTP session if not already created
        if self.session is None:
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self.config.api_key:  # ty: ignore[unresolved-attribute]
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),  # ty: ignore[unresolved-attribute]
                headers=headers,
            )

        try:
            # Test the connection
            if self.config.api_key:  # ty: ignore[unresolved-attribute]
                await self._test_connection()

            self._is_initialized = True
            logger.info("Perplexity provider initialized successfully")

        except Exception as e:
            error_msg = f"Failed to initialize Perplexity provider: {e!s}"
            logger.error(error_msg, exc_info=True)
            self.metrics["last_error"] = error_msg  # ty: ignore[invalid-assignment]
            raise RuntimeError(error_msg) from e

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if hasattr(self, "session"):
            await self.session.close()
        self._is_initialized = False
        logger.info("Perplexity provider cleaned up")

    async def list_models(self) -> list[dict[str, Any]]:
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
                "online": True,
            },
            {
                "id": "llama-3.1-sonar-small-128k-online",
                "name": "Llama 3.1 Sonar Small (Online)",
                "description": "Fast Perplexity model with web search",
                "capabilities": ["text-generation", "chat", "web-search"],
                "max_tokens": 4096,
                "context_length": 128000,
                "provider": "perplexity",
                "online": True,
            },
            {
                "id": "llama-3.1-sonar-large-128k-chat",
                "name": "Llama 3.1 Sonar Large (Offline)",
                "description": "Most capable Perplexity model without web search",
                "capabilities": ["text-generation", "chat"],
                "max_tokens": 4096,
                "context_length": 128000,
                "provider": "perplexity",
                "online": False,
            },
            {
                "id": "llama-3.1-sonar-small-128k-chat",
                "name": "Llama 3.1 Sonar Small (Offline)",
                "description": "Fast Perplexity model without web search",
                "capabilities": ["text-generation", "chat"],
                "max_tokens": 4096,
                "context_length": 128000,
                "provider": "perplexity",
                "online": False,
            },
        ]

        return models

    async def generate(self, prompt: str, model: str, **kwargs) -> AsyncGenerator[str, None]:  # ty: ignore[invalid-method-override]
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

        model_id = model or self.config.default_model  # ty: ignore[unresolved-attribute]
        start_time = time.time()
        self.metrics["total_requests"] += 1  # ty: ignore[unsupported-operator]

        try:
            # Prepare generation parameters
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),  # ty: ignore[unresolved-attribute]
                "temperature": kwargs.get("temperature", self.config.temperature),  # ty: ignore[unresolved-attribute]
                "top_p": kwargs.get("top_p", self.config.top_p),  # ty: ignore[unresolved-attribute]
                "top_k": kwargs.get("top_k", self.config.top_k),  # ty: ignore[unresolved-attribute]
                "stop": kwargs.get("stop", self.config.stop),  # ty: ignore[unresolved-attribute]
                "stream": True,
            }

            # Generate text using Perplexity streaming
            async with self.session.post(f"{self.config.base_url}/chat/completions", json=payload) as response:  # ty: ignore[unresolved-attribute]
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Perplexity API error: {response.status} - {error_text}")

                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            continue

            # Update metrics
            duration = time.time() - start_time
            self.metrics["successful_requests"] += 1  # ty: ignore[unsupported-operator]
            self.metrics["total_time_seconds"] += duration  # ty: ignore[unsupported-operator]

        except Exception as e:
            error_msg = f"Error in text generation: {e!s}"
            logger.error(error_msg, exc_info=True)
            self.metrics["failed_requests"] += 1  # ty: ignore[unsupported-operator]
            self.metrics["last_error"] = error_msg  # ty: ignore[invalid-assignment]
            raise RuntimeError(error_msg) from e

    async def chat_completion(self, messages: list[dict[str, str]], model: str | None = None, **kwargs) -> str:
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

        model_id = model or self.config.default_model  # ty: ignore[unresolved-attribute]
        start_time = time.time()
        self.metrics["total_requests"] += 1  # ty: ignore[unsupported-operator]

        try:
            # Prepare generation parameters
            payload = {
                "model": model_id,
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),  # ty: ignore[unresolved-attribute]
                "temperature": kwargs.get("temperature", self.config.temperature),  # ty: ignore[unresolved-attribute]
                "top_p": kwargs.get("top_p", self.config.top_p),  # ty: ignore[unresolved-attribute]
                "top_k": kwargs.get("top_k", self.config.top_k),  # ty: ignore[unresolved-attribute]
                "stop": kwargs.get("stop", self.config.stop),  # ty: ignore[unresolved-attribute]
                "stream": False,
            }

            # Generate response
            async with self.session.post(f"{self.config.base_url}/chat/completions", json=payload) as response:  # ty: ignore[unresolved-attribute]
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Perplexity API error: {response.status} - {error_text}")

                data = await response.json()
                response_text = data["choices"][0]["message"]["content"]

            # Update metrics
            duration = time.time() - start_time
            self.metrics["successful_requests"] += 1  # ty: ignore[unsupported-operator]
            self.metrics["total_tokens_generated"] += len(response_text.split())  # ty: ignore[unsupported-operator]
            self.metrics["total_time_seconds"] += duration  # ty: ignore[unsupported-operator]

            return response_text

        except Exception as e:
            error_msg = f"Error in chat completion: {e!s}"
            logger.error(error_msg, exc_info=True)
            self.metrics["failed_requests"] += 1  # ty: ignore[unsupported-operator]
            self.metrics["last_error"] = error_msg  # ty: ignore[invalid-assignment]
            raise RuntimeError(error_msg) from e

    async def pull_model(self, model_name: str) -> dict[str, Any]:
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

    async def get_metrics(self) -> dict[str, Any]:
        """Get provider metrics.

        Returns:
            Dictionary of metrics
        """
        metrics = self.metrics.copy()
        metrics.update(
            {
                "provider": "perplexity",
                "api_key_configured": self.config.api_key is not None,  # ty: ignore[unresolved-attribute]
                "base_url": self.config.base_url,  # ty: ignore[unresolved-attribute]
                "default_model": self.config.default_model,  # ty: ignore[unresolved-attribute]
            }
        )  # ty: ignore[no-matching-overload]

        return metrics

    async def health_check(self) -> dict[str, Any]:
        """Perform a health check of the provider.

        Returns:
            Health check results
        """
        status = {
            "status": "healthy" if self.is_ready else "unhealthy",
            "provider": "perplexity",
            "api_key_configured": self.config.api_key is not None,  # ty: ignore[unresolved-attribute]
            "last_error": self.metrics.get("last_error"),
            "total_requests": self.metrics["total_requests"],
            "successful_requests": self.metrics["successful_requests"],
            "failed_requests": self.metrics["failed_requests"],
        }

        # Test API connection if possible
        if self.config.api_key:  # ty: ignore[unresolved-attribute]
            try:
                await self._test_connection()
                status["api_connection"] = "healthy"
            except Exception as e:
                status["api_connection"] = "unhealthy"
                status["api_error"] = str(e)
        else:
            status["api_connection"] = "no_api_key"

        return status

    async def get_model_info(self, model_name: str) -> dict[str, Any]:
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

    async def get_model(self, model_id: str) -> ModelMetadata | None:  # ty: ignore[invalid-method-override]
        """Get details about a specific model."""
        models = await self.list_models()
        for model in models:
            if isinstance(model, dict) and model.get("id") == model_id:
                return ModelMetadata(
                    id=model["id"],
                    name=model["name"],
                    provider=ModelProvider.GEMINI if "gemini" in model.get("id", "") else ModelProvider.PERPLEXITY,
                    capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT],
                    parameters={"max_tokens": model.get("max_tokens", 4096)},
                )
            elif isinstance(model, ModelMetadata) and model.id == model_id:
                return model
        return None

    async def load_model(self, model_id: str, **kwargs) -> ModelMetadata:  # ty: ignore[invalid-method-override]
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

    async def chat(self, model_id: str, messages: list[dict[str, str]], **kwargs) -> str:
        """Generate a chat completion using the specified model."""
        response = await self.chat_completion(model_id=model_id, messages=messages, **kwargs)
        return response

    async def generate_embeddings(self, model_id: str, texts: list[str], **kwargs) -> list[list[float]]:
        """Generate embeddings for the given texts."""
        raise NotImplementedError("Embeddings not supported by this provider")

    async def _test_connection(self) -> None:
        """Test the connection to Perplexity API."""
        try:
            # Make a simple request to test the connection
            payload = {
                "model": "llama-3.1-sonar-small-128k-chat",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1,
            }

            async with self.session.post(f"{self.config.base_url}/chat/completions", json=payload) as response:  # ty: ignore[unresolved-attribute]
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API test failed: {response.status} - {error_text}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Perplexity API: {e!s}") from e
