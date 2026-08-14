"""Gemini provider implementation."""

import importlib
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from llm_mcp.models.base import BaseProvider, ModelCapability, ModelMetadata, ModelProvider, ModelStatus

logger = logging.getLogger(__name__)

# google.generativeai is optional — importlib keeps the name bound (Any).
genai: Any

try:
    genai = importlib.import_module("google.generativeai")
    GEMINI_AVAILABLE = True
except ImportError:
    logger.warning("Google Generative AI not installed. Install with: pip install google-generativeai")
    GEMINI_AVAILABLE = False
    genai = None


class GeminiProvider(BaseProvider):
    """
    Gemini provider for Google's Gemini models.

    Features:
    - Gemini 1.5 Pro, Flash
    - Streaming and non-streaming generation
    - Multimodal capabilities (text, images, audio)
    - Function calling support
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the Gemini provider.

        Args:
            config: Configuration dictionary for the Gemini provider
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Google Generative AI is not installed. Please install it with: pip install google-generativeai"
            )

        from .config import GeminiConfig

        self.config = GeminiConfig(**(config or {}))

        # Configure the generative AI
        if self.config.api_key:
            genai.configure(api_key=self.config.api_key)

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
        return "gemini"

    @property
    def is_ready(self) -> bool:
        """Check if the provider is ready to handle requests."""
        return self._is_initialized and self.config.api_key is not None  # ty: ignore[unresolved-attribute]

    async def initialize(self) -> None:
        """Initialize the Gemini provider."""
        if self._is_initialized:
            return

        logger.info("Initializing Gemini provider")

        try:
            # Test the connection
            if self.config.api_key:  # ty: ignore[unresolved-attribute]
                await self._test_connection()

            self._is_initialized = True
            logger.info("Gemini provider initialized successfully")

        except Exception as e:
            error_msg = f"Failed to initialize Gemini provider: {e!s}"
            logger.error(error_msg, exc_info=True)
            self.metrics["last_error"] = error_msg  # ty: ignore[invalid-assignment]
            raise RuntimeError(error_msg) from e

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self._is_initialized = False
        logger.info("Gemini provider cleaned up")

    async def list_models(self) -> list[dict[str, Any]]:
        """List available Gemini models.

        Returns:
            List of model information dictionaries
        """
        if not self.is_ready:
            await self.initialize()

        models = [
            {
                "id": "gemini-1.5-pro",
                "name": "Gemini 1.5 Pro",
                "description": "Most capable Gemini model with large context",
                "capabilities": ["text-generation", "chat", "vision", "audio"],
                "max_tokens": 8192,
                "context_length": 2000000,  # 2M tokens
                "provider": "gemini",
            },
            {
                "id": "gemini-1.5-flash",
                "name": "Gemini 1.5 Flash",
                "description": "Fast and efficient Gemini model",
                "capabilities": ["text-generation", "chat", "vision", "audio"],
                "max_tokens": 8192,
                "context_length": 1000000,  # 1M tokens
                "provider": "gemini",
            },
            {
                "id": "gemini-1.0-pro",
                "name": "Gemini 1.0 Pro",
                "description": "Previous generation Gemini Pro model",
                "capabilities": ["text-generation", "chat"],
                "max_tokens": 8192,
                "context_length": 30720,
                "provider": "gemini",
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
            # Get the model
            model_instance = genai.GenerativeModel(model_id)

            # Prepare generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),  # ty: ignore[unresolved-attribute]
                temperature=kwargs.get("temperature", self.config.temperature),  # ty: ignore[unresolved-attribute]
                top_p=kwargs.get("top_p", self.config.top_p),  # ty: ignore[unresolved-attribute]
                top_k=kwargs.get("top_k", self.config.top_k),  # ty: ignore[unresolved-attribute]
                stop_sequences=kwargs.get("stop_sequences", self.config.stop_sequences),  # ty: ignore[unresolved-attribute]
            )

            # Generate text using Gemini streaming
            response = await model_instance.generate_content_async(
                prompt, generation_config=generation_config, stream=True
            )

            async for chunk in response:
                if chunk.text:
                    yield chunk.text

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
            # Get the model
            model_instance = genai.GenerativeModel(model_id)

            # Prepare generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),  # ty: ignore[unresolved-attribute]
                temperature=kwargs.get("temperature", self.config.temperature),  # ty: ignore[unresolved-attribute]
                top_p=kwargs.get("top_p", self.config.top_p),  # ty: ignore[unresolved-attribute]
                top_k=kwargs.get("top_k", self.config.top_k),  # ty: ignore[unresolved-attribute]
                stop_sequences=kwargs.get("stop_sequences", self.config.stop_sequences),  # ty: ignore[unresolved-attribute]
            )

            # Convert messages to Gemini format
            chat = model_instance.start_chat(history=[])

            # Get the last user message
            user_message = None
            for message in reversed(messages):
                if message.get("role") == "user":
                    user_message = message.get("content", "")
                    break

            if not user_message:
                raise ValueError("No user message found in messages")

            # Generate response
            response = await chat.send_message_async(user_message, generation_config=generation_config)

            # Update metrics
            duration = time.time() - start_time
            self.metrics["successful_requests"] += 1  # ty: ignore[unsupported-operator]
            self.metrics["total_tokens_generated"] += len(response.text.split())  # ty: ignore[unsupported-operator]
            self.metrics["total_time_seconds"] += duration  # ty: ignore[unsupported-operator]

            return response.text

        except Exception as e:
            error_msg = f"Error in chat completion: {e!s}"
            logger.error(error_msg, exc_info=True)
            self.metrics["failed_requests"] += 1  # ty: ignore[unsupported-operator]
            self.metrics["last_error"] = error_msg  # ty: ignore[invalid-assignment]
            raise RuntimeError(error_msg) from e

    async def pull_model(self, model_name: str) -> dict[str, Any]:
        """Pull a model (not applicable for Gemini API).

        Args:
            model_name: Name of the model

        Returns:
            Model information
        """
        logger.info(f"Gemini models are API-based, no pulling needed for {model_name}")

        # Return model info from available models
        models = await self.list_models()
        for model in models:
            if model["id"] == model_name:
                return model

        raise ValueError(f"Model {model_name} not found in available Gemini models")

    async def get_metrics(self) -> dict[str, Any]:
        """Get provider metrics.

        Returns:
            Dictionary of metrics
        """
        metrics = self.metrics.copy()
        metrics.update(
            {
                "provider": "gemini",
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
            "provider": "gemini",
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

        raise ValueError(f"Model {model_name} not found in available Gemini models")

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
        """Test the connection to Gemini API."""
        try:
            # Make a simple request to test the connection
            model = genai.GenerativeModel("gemini-1.5-flash")
            await model.generate_content_async("Hi")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Gemini API: {e!s}") from e
