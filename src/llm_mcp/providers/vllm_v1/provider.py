"""vLLM V1 Provider with v1.0.0+ support and enhanced performance."""

import importlib
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from llm_mcp.models.base import BaseProvider

from .config import VLLMv1Config

logger = logging.getLogger(__name__)

# vLLM is an optional dependency. Import via importlib so the names stay
# bound (Any) whether or not it is importable — no try/except redeclaration.
LLM: Any
SamplingParams: Any
AsyncEngineArgs: Any
RequestOutput: Any
random_uuid: Any

try:
    _vllm = importlib.import_module("vllm")
    LLM = _vllm.LLM
    SamplingParams = _vllm.SamplingParams
    _arg_utils = importlib.import_module("vllm.engine.arg_utils")
    AsyncEngineArgs = _arg_utils.AsyncEngineArgs
    _outputs = importlib.import_module("vllm.outputs")
    RequestOutput = _outputs.RequestOutput
    _utils = importlib.import_module("vllm.utils")
    random_uuid = _utils.random_uuid
    VLLM_AVAILABLE = True
except ImportError:
    logger.warning("vLLM not installed. Install with: pip install vllm")
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None
    AsyncEngineArgs = None
    RequestOutput = None
    random_uuid = None


@dataclass
class VLLMGenerationResult:
    """Container for vLLM generation results."""

    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    metrics: dict[str, Any] | None = None


class VLLMv1Provider(BaseProvider):
    """
    vLLM V1 Provider with v1.0.0+ support.

    Features:
    - vLLM 1.0.0+ with PagedAttention v3
    - Continuous batching
    - Tensor parallelism
    - Multi-GPU support
    - Efficient KV cache management
    - Quantization support (AWQ, GPTQ, SqueezeLLM)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the vLLM provider.

        Args:
            config: Configuration dictionary for the vLLM provider
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed. Please install it with: pip install vllm")

        self.config = VLLMv1Config(**(config or {}))
        self.llm: Any = None  # vLLM LLM instance (Any: optional-dependency type)
        self.sampling_params = SamplingParams()
        self._is_initialized = False
        self._model_loaded = False
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

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
        return "vllm_v1"

    @property
    def is_ready(self) -> bool:
        """Check if the provider is ready to handle requests."""
        return self._is_initialized and self._model_loaded

    async def initialize(self) -> None:
        """Initialize the vLLM provider and load the model."""
        if self._is_initialized:
            return

        logger.info(f"Initializing vLLM provider with config: {self.config}")

        try:
            # Set up engine arguments
            engine_args = AsyncEngineArgs(
                model=self.config.model,  # ty: ignore[unknown-argument, unresolved-attribute]
                tokenizer=self.config.tokenizer,  # ty: ignore[unknown-argument, unresolved-attribute]
                tokenizer_mode=self.config.tokenizer_mode,  # ty: ignore[unknown-argument, unresolved-attribute]
                trust_remote_code=self.config.trust_remote_code,  # ty: ignore[unknown-argument, unresolved-attribute]
                download_dir=self.config.download_dir,  # ty: ignore[invalid-argument-type, unknown-argument, unresolved-attribute]
                tensor_parallel_size=self.config.tensor_parallel_size,  # ty: ignore[unknown-argument, unresolved-attribute]
                max_parallel_loading_workers=self.config.max_parallel_loading_workers,  # ty: ignore[unknown-argument, unresolved-attribute]
                block_size=self.config.block_size,  # ty: ignore[invalid-argument-type, unknown-argument, unresolved-attribute]
                use_v2_block_manager=self.config.use_v2_block_manager,  # ty: ignore[unknown-argument, unresolved-attribute]
                gpu_memory_utilization=self.config.gpu_memory_utilization,  # ty: ignore[unknown-argument, unresolved-attribute]
                swap_space=self.config.swap_space,  # ty: ignore[unknown-argument, unresolved-attribute]
                max_num_batched_tokens=self.config.max_num_batched_tokens,  # ty: ignore[unknown-argument, unresolved-attribute]
                max_num_seqs=self.config.max_num_seqs,  # ty: ignore[unknown-argument, unresolved-attribute]
                max_model_len=self.config.max_model_len,  # ty: ignore[unknown-argument, unresolved-attribute]
                seed=self.config.seed,  # ty: ignore[unknown-argument, unresolved-attribute]
                quantization=self.config.quantization,  # ty: ignore[unknown-argument, unresolved-attribute]
                enable_chunked_prefill=self.config.enable_chunked_prefill,  # ty: ignore[unknown-argument, unresolved-attribute]
            )

            # Initialize the LLM
            self.llm = LLM.from_engine_args(engine_args)  # ty: ignore[unresolved-attribute]
            self._model_loaded = True
            self._is_initialized = True

            logger.info(f"Successfully loaded model: {self.config.model}")  # ty: ignore[unresolved-attribute]

        except Exception as e:
            error_msg = f"Failed to initialize vLLM provider: {e!s}"
            logger.error(error_msg, exc_info=True)
            self.metrics["last_error"] = error_msg  # ty: ignore[invalid-assignment]
            raise RuntimeError(error_msg) from e
        self._load_supported_models()
        logger.info("vLLM V1 provider initialized")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if hasattr(self, "llm") and self.llm:
            if hasattr(self.llm.llm_engine, "shutdown"):  # ty: ignore[unresolved-attribute]
                self.llm.llm_engine.shutdown()  # ty: ignore[call-non-callable, unresolved-attribute]
            del self.llm
            self.llm = None
        self._is_initialized = False
        self._model_loaded = False
        logger.info("vLLM provider cleaned up")

    async def list_models(self) -> list[dict[str, Any]]:
        """List available models from the vLLM provider.

        Returns:
            List of model information dictionaries
        """
        if not self.is_ready:
            await self.initialize()

        # For vLLM, we typically work with a single model at a time
        # but we can return the currently loaded model's info
        if not self._model_loaded:
            return []

        model_info = {
            "id": self.config.model,  # ty: ignore[unresolved-attribute]
            "name": Path(self.config.model).name,  # ty: ignore[unresolved-attribute]
            "description": f"vLLM model: {self.config.model}",  # ty: ignore[unresolved-attribute]
            "capabilities": ["text-generation", "embeddings"],
            "max_length": getattr(
                self.llm.llm_engine.model_config,  # ty: ignore[unresolved-attribute]
                "max_model_len",
                self.config.max_seq_len,  # ty: ignore[unresolved-attribute]
            ),
            "device": self._device,
            "quantization": self.config.quantization,  # ty: ignore[unresolved-attribute]
            "tensor_parallel_size": self.config.tensor_parallel_size,  # ty: ignore[unresolved-attribute]
        }

        return [model_info]

    async def generate(self, prompt: str, model: str | None = None, **kwargs) -> AsyncGenerator[str, None]:  # ty: ignore[invalid-method-override]
        """Generate text from the model.

        Args:
            prompt: The input prompt
            model: Model to use (must match the loaded model)
            **kwargs: Additional generation parameters

        Yields:
            Chunks of generated text
        """
        if not self.is_ready:
            await self.initialize()

        if model and model != self.config.model:  # ty: ignore[unresolved-attribute]
            logger.warning(f"Requested model {model} doesn't match loaded model {self.config.model}")  # ty: ignore[unresolved-attribute]

        start_time = time.time()
        self.metrics["total_requests"] += 1  # ty: ignore[unsupported-operator]

        try:
            # Update sampling params from kwargs
            sampling_params = self._get_sampling_params(**kwargs)

            # Generate text using vLLM
            outputs = self.llm.generate(  # ty: ignore[unresolved-attribute]
                prompt,
                sampling_params=sampling_params,
                request_id=random_uuid(),
            )  # ty: ignore[no-matching-overload]

            # Stream the output
            full_text = ""
            for output in outputs:
                if not output.finished:
                    chunk = output.text[len(full_text) :]
                    full_text = output.text
                    yield chunk

            # Update metrics
            duration = time.time() - start_time
            self.metrics["successful_requests"] += 1  # ty: ignore[unsupported-operator]
            # Approximate token count
            self.metrics["total_tokens_generated"] += len(full_text.split())  # ty: ignore[unsupported-operator]
            self.metrics["total_time_seconds"] += duration  # ty: ignore[unsupported-operator]

        except Exception as e:
            error_msg = f"Error in text generation: {e!s}"
            logger.error(error_msg, exc_info=True)
            self.metrics["failed_requests"] += 1  # ty: ignore[unsupported-operator]
            self.metrics["last_error"] = error_msg  # ty: ignore[invalid-assignment]
            raise RuntimeError(error_msg) from e

    async def pull_model(self, model_name: str) -> dict[str, Any]:
        """Pull a model from the model hub.

        Args:
            model_name: Name of the model to pull

        Returns:
            Model information
        """
        logger.info(f"Pulling model: {model_name}")

        # If the model is already loaded, return its info
        if self._model_loaded and self.config.model == model_name:  # ty: ignore[unresolved-attribute]
            models = await self.list_models()
            if models:
                return models[0]

        # Clean up existing model if any
        if self.llm is not None:
            await self.cleanup()

        # Update config with new model
        self.config.model = model_name  # ty: ignore[invalid-assignment]

        try:
            # Re-initialize with new model
            await self.initialize()

            # Get model info
            model_info = await self.list_models()

            logger.info(f"Successfully pulled model: {model_name}")
            return model_info[0] if model_info else {}

        except Exception as e:
            error_msg = f"Failed to pull model {model_name}: {e!s}"
            logger.error(error_msg, exc_info=True)
            self.metrics["last_error"] = error_msg  # ty: ignore[invalid-assignment]
            raise RuntimeError(error_msg) from e

    def _get_sampling_params(self, **kwargs) -> Any:
        """Create SamplingParams from kwargs.

        Returns Any: SamplingParams is an optional-dependency type.
        """
        # Default values
        params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "max_tokens": 512,
            "stop": None,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        }

        # Update with any provided kwargs
        params.update({k: v for k, v in kwargs.items() if k in SamplingParams.__annotations__})

        return SamplingParams(**params)  # ty: ignore[invalid-argument-type]

    def get_metrics(self) -> dict[str, Any]:
        """Get provider metrics.

        Returns:
            Dictionary of metrics
        """
        metrics = self.metrics.copy()

        # Add vLLM specific metrics if available
        if hasattr(self, "llm") and self.llm is not None:
            metrics.update(
                {
                    "model_loaded": self._model_loaded,
                    "device": self._device,
                    "tensor_parallel_size": self.config.tensor_parallel_size,  # ty: ignore[unresolved-attribute]
                }
            )  # ty: ignore[no-matching-overload]

            # Add KV cache metrics if available
            if hasattr(self.llm.llm_engine, "cache_config"):  # ty: ignore[unresolved-attribute]
                cache_config = self.llm.llm_engine.cache_config  # ty: ignore[unresolved-attribute]
                metrics.update(
                    {
                        "cache_block_size": cache_config.block_size,
                        "gpu_memory_utilization": cache_config.gpu_memory_utilization,
                        "swap_space_gb": cache_config.swap_space_bytes / (1024**3)
                        if cache_config.swap_space_bytes
                        else 0,
                    }
                )

        return metrics

    def health_check(self) -> dict[str, Any]:
        """Perform a health check of the provider.

        Returns:
            Health check results
        """
        status = {
            "status": "healthy" if self.is_ready else "unhealthy",
            "model_loaded": self._model_loaded,
            "device_available": torch.cuda.is_available() if self._device == "cuda" else True,
            "last_error": self.metrics.get("last_error"),
            "total_requests": self.metrics["total_requests"],
            "successful_requests": self.metrics["successful_requests"],
            "failed_requests": self.metrics["failed_requests"],
        }

        # Add CUDA memory info if available
        if torch.cuda.is_available():
            status.update(
                {
                    "cuda_available": True,
                    "cuda_device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(),
                    "memory_allocated": torch.cuda.memory_allocated() / (1024**3),  # GB
                    "memory_reserved": torch.cuda.memory_reserved() / (1024**3),  # GB
                    "memory_free": torch.cuda.memory_reserved() - torch.cuda.memory_allocated() / (1024**3),  # GB
                }
            )

        return status

    def _load_supported_models(self) -> None:
        """Load the list of supported models."""
        # This would typically be loaded from a configuration file or API
        self._supported_models = {
            # Text models
            "meta-llama/Meta-Llama-3.1-8B-Instruct": {
                "id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "type": "text",
                "capabilities": ["text-generation", "chat"],
                "vram_required": "8GB",
                "context_length": 128000,
            },
            "meta-llama/Meta-Llama-3.1-70B-Instruct": {
                "id": "meta-llama/Meta-Llama-3.1-70B-Instruct",
                "type": "text",
                "capabilities": ["text-generation", "chat"],
                "vram_required": "40GB",
                "context_length": 128000,
                "tensor_parallel_recommended": 4,
            },
            # Multimodal models
            "Qwen/Qwen2-VL-7B-Instruct": {
                "id": "Qwen/Qwen2-VL-7B-Instruct",
                "type": "multimodal",
                "capabilities": ["text-generation", "chat", "vision"],
                "vram_required": "12GB",
                "context_length": 32768,
                "max_images": 20,
            },
            "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf": {
                "id": "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf",
                "type": "multimodal",
                "capabilities": ["text-generation", "chat", "vision"],
                "vram_required": "10GB",
                "context_length": 4096,
                "max_images": 10,
            },
            # Code models
            "deepseek-ai/deepseek-coder-33b-instruct": {
                "id": "deepseek-ai/deepseek-coder-33b-instruct",
                "type": "code",
                "capabilities": ["text-generation", "chat", "code-generation"],
                "vram_required": "20GB",
                "context_length": 16384,
            },
        }
