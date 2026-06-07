"""Service for managing LLM models and providers."""

import logging
from collections.abc import AsyncGenerator
from typing import Any

from ..config import Settings, get_settings
from ..providers import ProviderFactory
from ..utils.gpu import refresh_gpu_info
from .model_intelligence import ModelIntelligenceService

logger = logging.getLogger(__name__)


class ModelService:
    """Service for managing LLM models and providers."""

    def __init__(self, settings: Settings | None = None):
        """Initialize the model service.

        Args:
            settings: Application settings. If not provided, will use default settings.
        """
        self.settings = settings or get_settings()
        self.providers: dict[str, Any] = {}
        self.intelligence_service = ModelIntelligenceService()
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
                # Special handling for vLLM provider
                if provider_name.lower() in ("vllm", "vllm_v1"):
                    provider = await self._initialize_vllm_provider(provider_name, config)
                else:
                    provider = await ProviderFactory.create_and_verify_provider(provider_name, config)

                if provider:
                    self.providers[provider_name] = provider
                    logger.info(f"Initialized provider: {provider_name}")
                else:
                    logger.warning(f"Failed to initialize provider: {provider_name}")

            except Exception as e:
                logger.error(f"Error initializing provider {provider_name}: {e!s}", exc_info=True)

    async def _initialize_vllm_provider(self, provider_name: str, config: dict[str, Any]) -> Any:
        """Special initialization for vLLM provider.

        Args:
            provider_name: Name of the vLLM provider ('vllm' or 'vllm_v1')
            config: Provider configuration

        Returns:
            Initialized vLLM provider instance or None if initialization fails
        """
        try:
            # Import here to avoid circular imports
            from ..providers.vllm_v1.provider import VLLMv1Provider

            # Create provider instance
            provider = VLLMv1Provider(config)

            # Initialize the provider
            await provider.initialize()

            return provider

        except ImportError as e:
            logger.warning(f"vLLM provider not available: {e!s}")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize vLLM provider: {e!s}", exc_info=True)
            return None

    async def get_provider_for_model(self, model_name: str, provider_name: str | None = None) -> Any:
        """Get the appropriate provider for the given model.

        Args:
            model_name: Name of the model
            provider_name: Optional provider name if known

        Returns:
            A tuple of (provider, provider_name) or (None, None) if not found

        Raises:
            ValueError: If the provider is not available
        """
        # If provider is specified, use it
        if provider_name:
            provider = self.providers.get(provider_name.lower())
            if not provider:
                raise ValueError(f"Provider not found: {provider_name}")
            return provider, provider_name

        # Otherwise, try to find a provider that supports this model
        for name, provider in self.providers.items():
            try:
                models = await provider.list_models()
                if any(m.get("id") == model_name for m in models):
                    return provider, name
            except Exception as e:
                logger.warning(f"Error listing models for provider {name}: {e!s}")
                continue

        return None, None

        self._initialized = True
        logger.info("ModelService initialization complete")

    async def list_models(self, provider_name: str | None = None) -> list[dict[str, Any]]:
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
            gpu_data = refresh_gpu_info()
            available_vram = gpu_data["gpu"]["free_gb"] if gpu_data["available"] else 0

            for model in provider_models:
                model["provider"] = provider_name
                # Enrich with intelligence
                intel = self.intelligence_service.get_intelligence(model.get("id", ""))
                if intel:
                    model["intelligence"] = intel.model_dump()
                    model["hardware_compatibility"] = self.intelligence_service.get_compatibility(
                        intel.vram_required_gb or 0, available_vram
                    )
                models.append(model)
        else:
            # List models from all providers
            for provider_name, provider in self.providers.items():
                try:
                    provider_models = await provider.list_models()
                    gpu_data = refresh_gpu_info()
                    available_vram = gpu_data["gpu"]["free_gb"] if gpu_data["available"] else 0

                    for model in provider_models:
                        model["provider"] = provider_name
                        # Enrich with intelligence
                        intel = self.intelligence_service.get_intelligence(model.get("id", ""))
                        if intel:
                            model["intelligence"] = intel.model_dump()
                            model["hardware_compatibility"] = self.intelligence_service.get_compatibility(
                                intel.vram_required_gb or 0, available_vram
                            )
                        models.append(model)
                except Exception as e:
                    logger.error(f"Error listing models from provider {provider_name}: {e!s}", exc_info=True)

        return models

    async def generate(
        self,
        prompt: str,
        model: str,
        provider: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 100,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        images: list[str] | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Generate text using the specified model."""
        try:
            # Prepare common parameters
            params = {
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "stop": stop,
                "images": images,
                **kwargs,
            }

            # Special handling for vLLM provider
            if not provider or provider.lower() in ("vllm", "vllm_v1"):
                try:
                    from ..providers.vllm_v1.provider import VLLMv1Provider

                    # Get or create vLLM provider
                    vllm_provider = None
                    if "vllm" in self.providers:
                        vllm_provider = self.providers["vllm"]
                    elif "vllm_v1" in self.providers:
                        vllm_provider = self.providers["vllm_v1"]
                    else:
                        vllm_config = {
                            k: v
                            for k, v in kwargs.items()
                            if k
                            in [
                                "model",
                                "tensor_parallel_size",
                                "gpu_memory_utilization",
                                "max_seq_len",
                                "quantization",
                            ]
                        }
                        vllm_provider = VLLMv1Provider(vllm_config)
                        await vllm_provider.initialize()

                    # Generate using vLLM
                    async for chunk in vllm_provider.generate(prompt=prompt, model=model, **params):
                        yield chunk

                    return

                except ImportError as e:
                    if provider:
                        raise RuntimeError("vLLM provider is not available") from e
                except Exception as e:
                    logger.error(f"vLLM generation error: {e!s}", exc_info=True)
                    if provider:
                        raise RuntimeError(f"vLLM generation failed: {e!s}") from e

            # Standard provider interface
            provider_instance, _ = await self.get_provider_for_model(model, provider)
            if not provider_instance:
                raise ValueError(f"Model not found: {model}")

            async for chunk in provider_instance.generate(prompt=prompt, model=model, **params):
                yield chunk

        except Exception as e:
            logger.error(f"Generation error for {provider}/{model}: {e!s}", exc_info=True)
            raise

    async def get_model_info(self, model_name: str, provider_name: str | None = None) -> dict[str, Any]:
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
                raise ValueError(f"Error getting model info: {e!s}")

        # Otherwise, search all providers
        for provider_name, provider in self.providers.items():
            try:
                model_info = await provider.get_model_info(model_name)
                model_info["provider"] = provider_name

                # Enrich with intelligence
                intel = self.intelligence_service.get_intelligence(model_name)
                if intel:
                    model_info["intelligence"] = intel.model_dump()
                    gpu_data = refresh_gpu_info()
                    available_vram = gpu_data["gpu"]["free_gb"] if gpu_data["available"] else 0
                    model_info["hardware_compatibility"] = self.intelligence_service.get_compatibility(
                        intel.vram_required_gb or 0,
                        available_vram,
                    )

                return model_info
            except (ValueError, KeyError):
                continue
            except Exception as e:
                logger.warning(f"Error getting model info from {provider_name}: {e!s}")

        raise ValueError(f"Model not found: {model_name}")

    async def close(self) -> None:
        """Clean up resources."""
        for provider in self.providers.values():
            if hasattr(provider, "close") and callable(provider.close):
                await provider.close()

        self.providers.clear()
        self._initialized = False


# Create a singleton instance of the model service
model_service = ModelService()
