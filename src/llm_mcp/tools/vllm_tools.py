"""vLLM 0.9.5 integration for the LLM MCP server.

This module provides tools for working with vLLM 0.9.5, which offers
high-performance model serving with PagedAttention and continuous batching.

Key Features:
- PagedAttention for efficient memory usage
- Continuous batching for high throughput
- Tensor parallelism for multi-GPU support
- FlashAttention 2 optimization
"""
import asyncio
import json
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from dataclasses import dataclass
import structlog
import json

# vLLM 0.9.5 imports
try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.outputs import RequestOutput
    from vllm.utils import random_uuid
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

logger = structlog.get_logger(__name__)

try:
    # vLLM 1.0+ imports - UPDATED API
    from vllm import LLM, SamplingParams
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.outputs import RequestOutput
    from vllm.utils import random_uuid
    
    # Multimodal support (vLLM 1.0+ feature)
    try:
        from vllm.multimodal import MultiModalData
        MULTIMODAL_AVAILABLE = True
    except ImportError:
        MULTIMODAL_AVAILABLE = False
    
    VLLM_AVAILABLE = True
    logger.info("vLLM 1.0+ imports successful", multimodal=MULTIMODAL_AVAILABLE)
    
except ImportError as e:
@dataclass
class VLLMModelConfig:
    """Configuration for a vLLM 0.9.5 model."""
    model_name: str
    trust_remote_code: bool = False
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9
    swap_space: int = 4  # GB
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    quantization: Optional[str] = None
    enable_prefix_caching: bool = True

class VLLMManager:
    """Manager for vLLM 0.9.5 models."""
    
    def __init__(self):
        self.llm: Optional[LLM] = None
        self.current_model: Optional[str] = None
        self.model_config: Optional[VLLMModelConfig] = None
        self.sampling_params = SamplingParams()
        
        # Performance tracking
        self.total_requests = 0
        self.total_tokens_generated = 0
    
    def initialize_engine(self, model_config: VLLMModelConfig):
        """Initialize the vLLM 0.9.5 engine."""
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed. Please install it with: pip install vllm")
            
        self.model_config = model_config
        
        try:
            self.llm = LLM(
                model=model_config.model_name,
                trust_remote_code=model_config.trust_remote_code,
                max_model_len=model_config.max_model_len,
                gpu_memory_utilization=model_config.gpu_memory_utilization,
                swap_space=model_config.swap_space,
                tensor_parallel_size=model_config.tensor_parallel_size,
                dtype=model_config.dtype,
                quantization=model_config.quantization,
                enable_prefix_caching=model_config.enable_prefix_caching
            )
                
            self.current_model = model_config.model_name
            logger.info(f"Initialized vLLM engine with model: {self.current_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {str(e)}")
            raise
    
    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 100,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generate text using the current vLLM 0.9.5 model."""
        if not self.llm or not self.current_model:
            raise ValueError("vLLM engine not initialized. Call initialize_engine() first.")
            
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop or [],
            **kwargs
        )
        
        try:
            outputs = self.llm.generate(prompt, self.sampling_params)
            generated_text = outputs[0].outputs[0].text
            self.total_requests += 1
            self.total_tokens_generated += len(outputs[0].outputs[0].token_ids)
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise
    
    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        **generation_kwargs
    ) -> Dict[str, Any]:
        """Generate structured output (JSON) using vLLM 0.9.5."""
        if not self.llm or not self.current_model:
            raise ValueError("vLLM engine not initialized. Call initialize_engine() first.")
            
        # Add schema to the prompt for better guidance
        schema_prompt = f"""Generate a JSON output following this schema:
{json.dumps(schema, indent=2)}

Input: {prompt}

Output (JSON only, no markdown code blocks):"""
        
        try:
            result = self.generate_text(schema_prompt, **generation_kwargs)
            
            # Clean up the response to extract just the JSON
            if isinstance(result, str):
                # Try to extract JSON from markdown code blocks
                if '```json' in result:
                    result = result.split('```json')[1].split('```')[0].strip()
                elif '```' in result:
                    result = result.split('```')[1].strip()
                
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON response, returning as text")
                    return {"error": "Failed to parse JSON response", "raw_output": result}
            return result
            
        except Exception as e:
            logger.error(f"Error generating structured output: {str(e)}")
            return {"error": str(e)}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the current model."""
        stats = {
            "model": self.current_model or "No model loaded",
            "total_requests": self.total_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "avg_tokens_per_request": (
                self.total_tokens_generated / self.total_requests 
                if self.total_requests > 0 else 0
            )
        }
        
        # Add vLLM engine stats if available
        if self.llm and hasattr(self.llm.llm_engine, 'stats'):
            engine_stats = self.llm.llm_engine.stats
            stats.update({
                "engine_stats": {
                    "num_running": engine_stats.num_running,
                    "num_waiting": engine_stats.num_waiting,
                    "num_swapped": engine_stats.num_swapped,
                    "num_requests_finished": engine_stats.num_requests_finished,
                    "total_processing_time": engine_stats.total_processing_time,
                }
            })
            
        return stats
    
    def unload_model(self) -> None:
        """Unload the current model from memory."""
        if self.llm is not None:
            # vLLM 0.9.5 cleanup
            if hasattr(self.llm, '__del__'):
                del self.llm
            
            self.llm = None
            self.current_model = None
            self.model_config = None
            logger.info("vLLM model unloaded")

def register_vllm_tools(mcp):
    """Register vLLM 0.9.5 tools with the MCP server."""
    manager = VLLMManager()
    
    @mcp.tool("vllm_initialize")
    async def initialize_vllm_engine(
        model_name: str,
        trust_remote_code: bool = False,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        enable_prefix_caching: bool = True
    ) -> Dict[str, Any]:
        """Initialize the vLLM 0.9.5 engine with the specified configuration.
        
        Args:
            model_name: Name or path of the model to load
            trust_remote_code: Whether to trust remote code for model loading
            max_model_len: Maximum sequence length for the model
            gpu_memory_utilization: Fraction of GPU memory to use (0-1)
            swap_space: Amount of swap space to use in GB
            tensor_parallel_size: Number of GPUs for tensor parallelism
            dtype: Data type for model weights (auto, half, float16, bfloat16, float, float32)
            quantization: Quantization method to use (e.g., 'awq', 'squeezellm')
            enable_prefix_caching: Whether to enable prefix caching
            
        Returns:
            Dict with status and model information
        """
        try:
            config = VLLMModelConfig(
                model_name=model_name,
                trust_remote_code=trust_remote_code,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                swap_space=swap_space,
                tensor_parallel_size=tensor_parallel_size,
                dtype=dtype,
                quantization=quantization,
                enable_prefix_caching=enable_prefix_caching
            )
            
            manager.initialize_engine(config)
            
            return {
                "status": "success",
                "model": manager.current_model,
                "config": {
                    "max_model_len": max_model_len,
                    "gpu_memory_utilization": gpu_memory_utilization,
                    "tensor_parallel_size": tensor_parallel_size,
                    "dtype": dtype,
                    "quantization": quantization
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    @mcp.tool("vllm_generate")
    async def generate_text(
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 100,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text using the loaded vLLM 0.9.5 model."""
        try:
            result = manager.generate_text(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop,
                **kwargs
            )
            return {"status": "success", "text": result}
        except Exception as e:
            logger.error("vLLM generation failed", error=str(e))
            return {"status": "error", "error": str(e)}
    
    @mcp.tool("vllm_generate_structured")
    async def generate_structured(
        prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """Generate structured JSON output using vLLM 0.9.5."""
        try:
            result = manager.generate_structured(
                prompt=prompt,
                schema=schema,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return {"status": "success", **result}
        except Exception as e:
            logger.error("vLLM structured generation failed", error=str(e))
            return {"status": "error", "error": str(e)}
    
    @mcp.tool("vllm_stats")
    async def get_stats() -> Dict[str, Any]:
        """Get statistics for the vLLM 0.9.5 engine."""
        try:
            return {
                "status": "success",
                "stats": manager.get_performance_stats()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    @mcp.tool("vllm_unload")
    async def unload_model() -> Dict[str, str]:
        """Unload the current vLLM 0.9.5 model from memory."""
        try:
            manager.unload_model()
            return {"status": "success", "message": "Model unloaded"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    # Cleanup on server shutdown
    @mcp.on_shutdown
    async def cleanup():
        if hasattr(manager, 'llm') and manager.llm is not None:
            manager.unload_model()
            await vllm_manager.unload_model()
    
    return {"vllm_available": True, "tools_registered": 5}
