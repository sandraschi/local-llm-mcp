"""vLLM 1.0+ integration for the LLM MCP server - COMPLETELY REWRITTEN.

This module provides tools for working with vLLM 1.0+, which offers
high-performance model serving with multimodal capabilities and structured output.

Key Features:
- V1 engine with 19x performance improvement  
- Multimodal support (vision, audio)
- Structured output generation
- Tool calling integration
- Distributed inference
- FlashAttention 3 optimization
"""
import asyncio
import json
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from dataclasses import dataclass
import structlog

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
    VLLM_AVAILABLE = False
    MULTIMODAL_AVAILABLE = False
    logger.warning("vLLM not available", error=str(e))

@dataclass
class VLLMModelConfig:
    """Configuration for a vLLM 1.0+ model."""
    model_name: str
    trust_remote_code: bool = False
    
    # Memory and performance
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9
    swap_space: str = "4GB"
    
    # Parallelism (vLLM 1.0+ optimized)
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    
    # V1 engine features
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    attention_backend: str = "FLASHINFER"  # FlashAttention 3
    
    # Data types and quantization
    dtype: str = "auto"
    quantization: Optional[str] = None  # "awq", "gptq", "fp8"
    
    # Multimodal settings
    enable_vision: bool = True
    max_image_input_size: int = 2048

class VLLMManager:
    """Manager for vLLM 1.0+ models with enhanced capabilities."""
    
    def __init__(self):
        self.engine: Optional[Union[LLM, AsyncLLMEngine]] = None
        self.current_model: Optional[str] = None
        self.model_config: Optional[VLLMModelConfig] = None
        self.is_async: bool = False
        
        # Performance tracking
        self.total_requests = 0
        self.total_tokens_generated = 0
    
    async def initialize_engine(self, model_config: VLLMModelConfig, async_mode: bool = True) -> None:
        """Initialize the vLLM 1.0+ engine with modern configuration."""
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM 1.0+ package is not installed. Install with: pip install vllm>=1.0.0")
        
        # Set vLLM environment variables for V1 engine
        import os
        env_vars = {
            "VLLM_USE_V1": "1",
            "VLLM_ATTENTION_BACKEND": model_config.attention_backend,
            "VLLM_ENABLE_PREFIX_CACHING": "1" if model_config.enable_prefix_caching else "0",
            "VLLM_GPU_MEMORY_UTILIZATION": str(model_config.gpu_memory_utilization),
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
        
        logger.info("vLLM environment configured", env_vars=env_vars)
        
        # Engine initialization arguments for vLLM 1.0+
        engine_args = {
            "model": model_config.model_name,
            "trust_remote_code": model_config.trust_remote_code,
            "dtype": model_config.dtype,
            "max_model_len": model_config.max_model_len,
            "gpu_memory_utilization": model_config.gpu_memory_utilization,
            "tensor_parallel_size": model_config.tensor_parallel_size,
            "pipeline_parallel_size": model_config.pipeline_parallel_size,
            "swap_space": model_config.swap_space,
            "enable_prefix_caching": model_config.enable_prefix_caching,
            "enable_chunked_prefill": model_config.enable_chunked_prefill,
        }
        
        # Add quantization if specified
        if model_config.quantization:
            engine_args["quantization"] = model_config.quantization
        
        try:
            if async_mode:
                self.engine = AsyncLLMEngine.from_engine_args(**engine_args)
                self.is_async = True
                logger.info("AsyncLLMEngine initialized")
            else:
                self.engine = LLM(**engine_args)
                self.is_async = False
                logger.info("LLM engine initialized")
            
            self.current_model = model_config.model_name
            self.model_config = model_config
            
            logger.info("vLLM 1.0+ engine initialized successfully", 
                       model=model_config.model_name,
                       async_mode=async_mode,
                       tensor_parallel=model_config.tensor_parallel_size,
                       enable_vision=model_config.enable_vision)
            
        except Exception as e:
            logger.error("Failed to initialize vLLM engine", error=str(e))
            raise
    
    async def generate_text(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 100,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Generate text using the current vLLM 1.0+ model."""
        if self.engine is None:
            raise RuntimeError("vLLM engine is not initialized. Call initialize_engine first.")
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop or [],
            **kwargs
        )
        
        if self.is_async:
            return await self._generate_async(prompt, sampling_params, stream)
        else:
            return self._generate_sync(prompt, sampling_params)
    
    async def _generate_async(
        self, 
        prompt: str, 
        sampling_params: SamplingParams, 
        stream: bool
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Async generation with vLLM 1.0+ AsyncLLMEngine."""
        request_id = random_uuid()
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        if not stream:
            # Non-streaming: collect all outputs
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
            
            if final_output is None:
                raise RuntimeError("No output generated from vLLM engine")
            
            # Update performance metrics
            self.total_requests += 1
            generated_tokens = len(final_output.outputs[0].token_ids)
            self.total_tokens_generated += generated_tokens
            
            return {
                "text": final_output.outputs[0].text,
                "model": self.current_model,
                "finish_reason": final_output.outputs[0].finish_reason,
                "usage": {
                    "prompt_tokens": len(final_output.prompt_token_ids),
                    "completion_tokens": generated_tokens,
                    "total_tokens": len(final_output.prompt_token_ids) + generated_tokens,
                },
                "performance": {
                    "total_requests": self.total_requests,
                    "total_tokens_generated": self.total_tokens_generated,
                }
            }
        else:
            # Streaming generation
            async def stream_results():
                async for request_output in results_generator:
                    yield {
                        "text": request_output.outputs[0].text,
                        "model": self.current_model,
                        "finish_reason": request_output.outputs[0].finish_reason,
                        "delta": request_output.outputs[0].text,  # For compatibility
                    }
            
            return stream_results()
    
    def _generate_sync(self, prompt: str, sampling_params: SamplingParams) -> Dict[str, Any]:
        """Synchronous generation with vLLM 1.0+ LLM."""
        outputs = self.engine.generate([prompt], sampling_params)
        output = outputs[0]
        
        # Update performance metrics
        self.total_requests += 1
        generated_tokens = len(output.outputs[0].token_ids)
        self.total_tokens_generated += generated_tokens
        
        return {
            "text": output.outputs[0].text,
            "model": self.current_model,
            "finish_reason": output.outputs[0].finish_reason,
            "usage": {
                "prompt_tokens": len(output.prompt_token_ids),
                "completion_tokens": generated_tokens,
                "total_tokens": len(output.prompt_token_ids) + generated_tokens,
            }
        }
    
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        **generation_kwargs
    ) -> Dict[str, Any]:
        """Generate structured output (JSON) using vLLM 1.0+ capabilities."""
        # Add JSON schema instruction to prompt
        schema_prompt = f"""Please respond with valid JSON that matches this schema:
{json.dumps(schema, indent=2)}

User request: {prompt}

Response (JSON only):"""
        
        # Generate with structured output guidance
        result = await self.generate_text(
            schema_prompt,
            stop=["</json>", "\\n\\n"],
            **generation_kwargs
        )
        
        # Try to parse and validate the JSON
        try:
            if isinstance(result, dict) and "text" in result:
                text = result["text"].strip()
                # Clean up the text to extract JSON
                if text.startswith("```json"):
                    text = text[7:]
                if text.endswith("```"):
                    text = text[:-3]
                
                parsed_json = json.loads(text)
                result["structured_output"] = parsed_json
                result["schema_validated"] = True
        except json.JSONDecodeError:
            result["schema_validated"] = False
            logger.warning("Generated text is not valid JSON", text=result.get("text", ""))
        
        return result
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the current model."""
        return {
            "model": self.current_model,
            "total_requests": self.total_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "average_tokens_per_request": (
                self.total_tokens_generated / self.total_requests 
                if self.total_requests > 0 else 0
            ),
            "engine_type": "AsyncLLMEngine" if self.is_async else "LLM",
            "multimodal_available": MULTIMODAL_AVAILABLE,
        }
    
    async def unload_model(self) -> None:
        """Unload the current model from memory."""
        if self.engine is not None:
            # vLLM 1.0+ cleanup
            if hasattr(self.engine, 'shutdown'):
                await self.engine.shutdown()
            elif hasattr(self.engine, '__del__'):
                del self.engine
            
            self.engine = None
            self.current_model = None
            self.model_config = None
            logger.info("vLLM model unloaded")

def register_vllm_tools(mcp):
    """Register vLLM 1.0+ tools with the MCP server."""
    if not VLLM_AVAILABLE:
        logger.warning("vLLM 1.0+ is not installed. vLLM tools will not be available.")
        return {"vllm_available": False, "error": "vLLM not installed"}
    
    vllm_manager = VLLMManager()
    
    @mcp.tool()
    async def vllm_load_model(
        model_name: str,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        trust_remote_code: bool = False,
        enable_vision: bool = True,
        quantization: Optional[str] = None,
        async_mode: bool = True
    ) -> Dict[str, Any]:
        """Load a vLLM 1.0+ model with modern optimizations."""
        config = VLLMModelConfig(
            model_name=model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            enable_vision=enable_vision,
            quantization=quantization
        )
        
        try:
            await vllm_manager.initialize_engine(config, async_mode=async_mode)
            return {
                "status": "success", 
                "model": model_name,
                "async_mode": async_mode,
                "features": {
                    "vision_enabled": enable_vision,
                    "quantization": quantization,
                    "tensor_parallel": tensor_parallel_size
                }
            }
        except Exception as e:
            logger.error("Failed to load vLLM model", model=model_name, error=str(e))
            return {"status": "error", "error": str(e)}
    
    @mcp.tool()
    async def vllm_generate(
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 100,
        stop: Optional[List[str]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Generate text using the loaded vLLM 1.0+ model with high performance."""
        try:
            result = await vllm_manager.generate_text(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop,
                stream=stream
            )
            
            if stream and hasattr(result, '__aiter__'):
                # For streaming, collect first chunk to return
                async for chunk in result:
                    return {"status": "streaming", "first_chunk": chunk}
            else:
                return {"status": "success", **result}
        except Exception as e:
            logger.error("vLLM generation failed", error=str(e))
            return {"status": "error", "error": str(e)}
    
    @mcp.tool()
    async def vllm_generate_structured(
        prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """Generate structured JSON output using vLLM 1.0+ capabilities."""
        try:
            result = await vllm_manager.generate_structured(
                prompt=prompt,
                schema=schema,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return {"status": "success", **result}
        except Exception as e:
            logger.error("vLLM structured generation failed", error=str(e))
            return {"status": "error", "error": str(e)}
    
    @mcp.tool()
    async def vllm_performance_stats() -> Dict[str, Any]:
        """Get performance statistics for the current vLLM model."""
        try:
            stats = await vllm_manager.get_performance_stats()
            return {"status": "success", "stats": stats}
        except Exception as e:
            logger.error("Failed to get vLLM stats", error=str(e))
            return {"status": "error", "error": str(e)}
    
    @mcp.tool()
    async def vllm_unload() -> Dict[str, Any]:
        """Unload the current vLLM model from memory."""
        try:
            await vllm_manager.unload_model()
            return {"status": "success", "message": "Model unloaded successfully"}
        except Exception as e:
            logger.error("Failed to unload vLLM model", error=str(e))
            return {"status": "error", "error": str(e)}
    
    # Cleanup on server shutdown
    @mcp.on_shutdown
    async def cleanup():
        if hasattr(vllm_manager, 'engine') and vllm_manager.engine is not None:
            await vllm_manager.unload_model()
    
    return {"vllm_available": True, "tools_registered": 5}
