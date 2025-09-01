"""vLLM V1 Provider with v1.0.0+ support and enhanced performance."""

import os
import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from pathlib import Path
from dataclasses import dataclass

import aiohttp
import torch
from pydantic import BaseModel, Field

from ..base import BaseProvider
from .config import VLLMv1Config

logger = logging.getLogger(__name__)

# Try to import vLLM, but make it optional
try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.outputs import RequestOutput
    from vllm.utils import random_uuid
    VLLM_AVAILABLE = True
except ImportError:
    logger.warning("vLLM not installed. Install with: pip install vllm")
    VLLM_AVAILABLE = False

@dataclass
class VLLMGenerationResult:
    """Container for vLLM generation results."""
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    metrics: Dict[str, Any] = None

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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the vLLM provider.
        
        Args:
            config: Configuration dictionary for the vLLM provider
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed. Please install it with: pip install vllm")
            
        self.config = VLLMv1Config(**(config or {}))
        self.llm: Optional[LLM] = None
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
            "last_error": None
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
                model=self.config.model,
                tokenizer=self.config.tokenizer,
                tokenizer_mode=self.config.tokenizer_mode,
                trust_remote_code=self.config.trust_remote_code,
                download_dir=self.config.download_dir,
                tensor_parallel_size=self.config.tensor_parallel_size,
                max_parallel_loading_workers=self.config.max_parallel_loading_workers,
                block_size=self.config.block_size,
                use_v2_block_manager=self.config.use_v2_block_manager,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                swap_space=self.config.swap_space,
                max_num_batched_tokens=self.config.max_num_batched_tokens,
                max_num_seqs=self.config.max_num_seqs,
                max_model_len=self.config.max_model_len,
                seed=self.config.seed,
                quantization=self.config.quantization,
                enable_chunked_prefill=self.config.enable_chunked_prefill,
            )
            
            # Initialize the LLM
            self.llm = LLM.from_engine_args(engine_args)
            self._model_loaded = True
            self._is_initialized = True
            
            logger.info(f"Successfully loaded model: {self.config.model}")
            
        except Exception as e:
            error_msg = f"Failed to initialize vLLM provider: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.metrics["last_error"] = error_msg
            raise RuntimeError(error_msg) from e
        await self._load_supported_models()
        logger.info("vLLM V1 provider initialized")
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if hasattr(self, 'llm') and self.llm:
            if hasattr(self.llm.llm_engine, 'shutdown'):
                self.llm.llm_engine.shutdown()
            del self.llm
            self.llm = None
        self._is_initialized = False
        self._model_loaded = False
        logger.info("vLLM provider cleaned up")
    
    async def list_models(self) -> List[Dict[str, Any]]:
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
            "id": self.config.model,
            "name": Path(self.config.model).name,
            "description": f"vLLM model: {self.config.model}",
            "capabilities": ["text-generation", "embeddings"],
            "max_length": getattr(self.llm.llm_engine.model_config, "max_model_len", self.config.max_seq_len),
            "device": self._device,
            "quantization": self.config.quantization,
            "tensor_parallel_size": self.config.tensor_parallel_size
        }
        
        return [model_info]
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
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
            
        if model and model != self.config.model:
            logger.warning(f"Requested model {model} doesn't match loaded model {self.config.model}")
            
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        try:
            # Update sampling params from kwargs
            sampling_params = self._get_sampling_params(**kwargs)
            
            # Generate text using vLLM
            outputs = self.llm.generate(
                prompt,
                sampling_params=sampling_params,
                request_id=random_uuid(),
            )
            
            # Stream the output
            full_text = ""
            for output in outputs:
                if not output.finished:
                    chunk = output.text[len(full_text):]
                    full_text = output.text
                    yield chunk
            
            # Update metrics
            duration = time.time() - start_time
            self.metrics["successful_requests"] += 1
            self.metrics["total_tokens_generated"] += len(full_text.split())  # Approximate
            self.metrics["total_time_seconds"] += duration
            
        except Exception as e:
            error_msg = f"Error in text generation: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.metrics["failed_requests"] += 1
            self.metrics["last_error"] = error_msg
            raise RuntimeError(error_msg) from e
    
    async def pull_model(self, model_name: str) -> Dict[str, Any]:
        """Pull a model from the model hub.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            Model information
        """
        logger.info(f"Pulling model: {model_name}")
        
        # If the model is already loaded, return its info
        if self._model_loaded and self.config.model == model_name:
            return await self.list_models()[0]
            
        # Clean up existing model if any
        if self.llm is not None:
            await self.cleanup()
            
        # Update config with new model
        self.config.model = model_name
        
        try:
            # Re-initialize with new model
            await self.initialize()
            
            # Get model info
            model_info = await self.list_models()
            
            logger.info(f"Successfully pulled model: {model_name}")
            return model_info[0] if model_info else {}
            
        except Exception as e:
            error_msg = f"Failed to pull model {model_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.metrics["last_error"] = error_msg
            raise RuntimeError(error_msg) from e
    
    def _get_sampling_params(self, **kwargs) -> SamplingParams:
        """Create SamplingParams from kwargs."""
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
        
        return SamplingParams(**params)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get provider metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = self.metrics.copy()
        
        # Add vLLM specific metrics if available
        if hasattr(self, 'llm') and self.llm is not None:
            metrics.update({
                "model_loaded": self._model_loaded,
                "device": self._device,
                "tensor_parallel_size": self.config.tensor_parallel_size,
            })
            
            # Add KV cache metrics if available
            if hasattr(self.llm.llm_engine, 'cache_config'):
                cache_config = self.llm.llm_engine.cache_config
                metrics.update({
                    "cache_block_size": cache_config.block_size,
                    "gpu_memory_utilization": cache_config.gpu_memory_utilization,
                    "swap_space_gb": cache_config.swap_space_bytes / (1024**3) if cache_config.swap_space_bytes else 0,
                })
                
        return metrics
    
    async def health_check(self) -> Dict[str, Any]:
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
            status.update({
                "cuda_available": True,
                "cuda_device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(),
                "memory_allocated": torch.cuda.memory_allocated() / (1024**3),  # GB
                "memory_reserved": torch.cuda.memory_reserved() / (1024**3),    # GB
                "memory_free": torch.cuda.memory_reserved() - torch.cuda.memory_allocated() / (1024**3),  # GB
            })
            
        return status
    async def start_server(
        self, 
        model_id: str,
        tensor_parallel: int = 1,
        pipeline_parallel: int = 1,
        **kwargs
    ) -> None:
        """Start vLLM V1 server with specified model."""
        if self.is_ready:
            logger.info("vLLM V1 server already running")
            return
        
        # Build command for vLLM V1 server
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_id,
            "--host", self.config.host,
            "--port", str(self.config.port),
            "--tensor-parallel-size", str(tensor_parallel),
        ]
        
        # V1 engine configuration
        env = os.environ.copy()
        env.update({
            "VLLM_USE_V1": "1",  # Enable V1 engine
            "VLLM_ATTENTION_BACKEND": "FLASHINFER",  # FlashAttention 3
            "VLLM_GPU_MEMORY_UTILIZATION": str(self.config.gpu_memory_utilization),
        })
        
        # Add multimodal configuration if needed
        if self._is_multimodal_model(model_id):
            cmd.extend([
                "--enable-prefix-caching",  # Essential for multimodal
                "--max-model-len", str(self.config.max_seq_len),
            ])
        
        # Add distributed configuration
        if tensor_parallel > 1:
            cmd.extend(["--tensor-parallel-size", str(tensor_parallel)])
        if pipeline_parallel > 1:
            cmd.extend(["--pipeline-parallel-size", str(pipeline_parallel)])
        
        logger.info(f"Starting vLLM V1 server with model: {model_id}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Start the server process
        self.server_process = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Wait for server to be ready
        await self._wait_for_server_ready()
        self.current_model = model_id
        logger.info(f"vLLM V1 server started successfully with {model_id}")
    
    async def stop_server(self) -> None:
        """Stop the vLLM V1 server."""
        if self.server_process:
            self.server_process.terminate()
            await self.server_process.wait()
            self.server_process = None
            self.current_model = None
            logger.info("vLLM V1 server stopped")
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all supported models."""
        if not self.is_ready:
            return list(self._supported_models.values())
        
        try:
            async with self.session.get(f"{self.base_url}/v1/models") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
                else:
                    logger.warning(f"Failed to list models: {response.status}")
                    return list(self._supported_models.values())
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return list(self._supported_models.values())
    
    async def generate_text(
        self, 
        model_id: str, 
        prompt: str, 
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text using vLLM V1."""
        if not self.is_ready or self.current_model != model_id:
            await self.start_server(model_id)
        
        payload = {
            "model": model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            **kwargs
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/v1/completions",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["text"]
                else:
                    error_text = await response.text()
                    raise Exception(f"vLLM V1 error: {response.status} - {error_text}")
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    async def chat_completion(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate chat completion using vLLM V1."""
        if not self.is_ready or self.current_model != model_id:
            await self.start_server(model_id)
        
        payload = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            **kwargs
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    raise Exception(f"vLLM V1 error: {response.status} - {error_text}")
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise
    
    async def generate_with_vision(
        self,
        model_id: str,
        prompt: str,
        images: List[str],
        max_tokens: int = 512,
        **kwargs
    ) -> str:
        """Generate response with vision input using multimodal models."""
        if not self._is_multimodal_model(model_id):
            raise ValueError(f"Model {model_id} does not support vision")
        
        if not self.is_ready or self.current_model != model_id:
            await self.start_server(model_id)
        
        # Format message with images
        content = [{"type": "text", "text": prompt}]
        for image in images:
            if image.startswith(("http://", "https://")):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image}
                })
            else:
                # Local file - convert to base64
                import base64
                with open(image, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode()
                content.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                })
        
        messages = [{"role": "user", "content": content}]
        
        return await self.chat_completion(
            model_id=model_id,
            messages=messages,
            max_tokens=max_tokens,
            **kwargs
        )
    
    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a model."""
        model_info = self._supported_models.get(model_id, {})
        
        if self.is_ready:
            try:
                async with self.session.get(f"{self.base_url}/v1/models/{model_id}") as response:
                    if response.status == 200:
                        return await response.json()
            except Exception as e:
                logger.warning(f"Could not fetch live model info: {e}")
        
        return model_info
    
    async def benchmark_performance(
        self, 
        model_id: str, 
        test_prompts: List[str]
    ) -> Dict[str, Any]:
        """Benchmark vLLM V1 performance."""
        import time
        
        if not self.is_ready or self.current_model != model_id:
            await self.start_server(model_id)
        
        results = {
            "model": model_id,
            "provider": "vllm_v1",
            "test_count": len(test_prompts),
            "results": []
        }
        
        total_tokens = 0
        total_time = 0
        
        for i, prompt in enumerate(test_prompts):
            start_time = time.time()
            try:
                response = await self.generate_text(
                    model_id=model_id,
                    prompt=prompt,
                    max_tokens=100
                )
                end_time = time.time()
                
                duration = end_time - start_time
                tokens = len(response.split())  # Rough token estimate
                
                results["results"].append({
                    "prompt_index": i,
                    "duration_seconds": duration,
                    "tokens_generated": tokens,
                    "tokens_per_second": tokens / duration if duration > 0 else 0,
                    "success": True
                })
                
                total_tokens += tokens
                total_time += duration
                
            except Exception as e:
                results["results"].append({
                    "prompt_index": i,
                    "error": str(e),
                    "success": False
                })
        
        # Calculate aggregate metrics
        if total_time > 0:
            results["average_tokens_per_second"] = total_tokens / total_time
            results["total_duration"] = total_time
            results["total_tokens"] = total_tokens
        
        return results
    
    def _is_multimodal_model(self, model_id: str) -> bool:
        """Check if model supports multimodal input."""
        multimodal_patterns = [
            "qwen2-vl", "llava", "llava-next", "llava-onevision",
            "cogvlm", "blip", "instructblip", "minigpt",
            "flamingo", "kosmos", "git", "vision"
        ]
        return any(pattern in model_id.lower() for pattern in multimodal_patterns)
    
    async def _wait_for_server_ready(self, timeout: int = 60) -> None:
        """Wait for the vLLM V1 server to be ready."""
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            try:
                if not self.session:
                    self.session = aiohttp.ClientSession()
                
                async with self.session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        logger.info("vLLM V1 server is ready")
                        return
            except Exception:
                pass
            
            await asyncio.sleep(2)
        
        raise TimeoutError(f"vLLM V1 server did not become ready within {timeout} seconds")
    
    async def _load_supported_models(self) -> None:
        """Load the list of supported models."""
        # This would typically be loaded from a configuration file or API
        self._supported_models = {
            # Text models
            "meta-llama/Meta-Llama-3.1-8B-Instruct": {
                "id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "type": "text",
                "capabilities": ["text-generation", "chat"],
                "vram_required": "8GB",
                "context_length": 128000
            },
            "meta-llama/Meta-Llama-3.1-70B-Instruct": {
                "id": "meta-llama/Meta-Llama-3.1-70B-Instruct", 
                "type": "text",
                "capabilities": ["text-generation", "chat"],
                "vram_required": "40GB",
                "context_length": 128000,
                "tensor_parallel_recommended": 4
            },
            
            # Multimodal models
            "Qwen/Qwen2-VL-7B-Instruct": {
                "id": "Qwen/Qwen2-VL-7B-Instruct",
                "type": "multimodal",
                "capabilities": ["text-generation", "chat", "vision"],
                "vram_required": "12GB",
                "context_length": 32768,
                "max_images": 20
            },
            "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf": {
                "id": "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf",
                "type": "multimodal", 
                "capabilities": ["text-generation", "chat", "vision"],
                "vram_required": "10GB",
                "context_length": 4096,
                "max_images": 10
            },
            
            # Code models
            "deepseek-ai/deepseek-coder-33b-instruct": {
                "id": "deepseek-ai/deepseek-coder-33b-instruct",
                "type": "code",
                "capabilities": ["text-generation", "chat", "code-generation"],
                "vram_required": "20GB", 
                "context_length": 16384
            }
        }
