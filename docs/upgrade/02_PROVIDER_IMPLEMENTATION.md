# vLLM V1 Provider Implementation

## Step 1: Model Configuration System

**File: `src/llm_mcp/providers/vllm_v1/__init__.py`**
```python
"""vLLM V1 Provider Package"""

from .provider import VLLMv1Provider
from .models import VLLMModelConfig, get_optimal_model_config
from .utils import VLLMHealthCheck, estimate_memory_usage

__all__ = [
    "VLLMv1Provider",
    "VLLMModelConfig", 
    "get_optimal_model_config",
    "VLLMHealthCheck",
    "estimate_memory_usage",
]
```

**File: `src/llm_mcp/providers/vllm_v1/models.py`**
```python
"""RTX 4090 Optimized Model Configurations"""

from typing import Dict, Optional, List
from pydantic import BaseModel
import psutil
import torch

class VLLMModelConfig(BaseModel):
    """vLLM V1 model configuration"""
    model_id: str
    model_type: str  # "text", "vision", "audio"
    vram_required: int  # GB
    ram_required: int   # GB
    tensor_parallel_size: int = 1
    max_seq_len: int = 4096
    gpu_memory_utilization: float = 0.85
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    trust_remote_code: bool = True
    revision: Optional[str] = None
    expected_tokens_per_sec: int = 0

# RTX 4090 24GB + 64GB RAM Optimized Models
RTX_4090_MODELS = {
    # Fast Models for RTX 4090
    "meta-llama/Llama-3.1-8B-Instruct": VLLMModelConfig(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        model_type="text",
        vram_required=16,
        ram_required=8,
        max_seq_len=8192,
        expected_tokens_per_sec=200,
    ),
    
    "microsoft/Phi-3-mini-4k-instruct": VLLMModelConfig(
        model_id="microsoft/Phi-3-mini-4k-instruct",
        model_type="text",
        vram_required=8,
        ram_required=4,
        max_seq_len=4096,
        gpu_memory_utilization=0.60,
        expected_tokens_per_sec=300,
    ),
    
    # Code Models
    "codellama/CodeLlama-13b-Instruct-hf": VLLMModelConfig(
        model_id="codellama/CodeLlama-13b-Instruct-hf",
        model_type="text",
        vram_required=20,
        ram_required=12,
        max_seq_len=16384,
        expected_tokens_per_sec=120,
    ),
    
    # Vision Models
    "Qwen/Qwen2-VL-7B-Instruct": VLLMModelConfig(
        model_id="Qwen/Qwen2-VL-7B-Instruct",
        model_type="vision",
        vram_required=18,
        ram_required=10,
        max_seq_len=4096,
        trust_remote_code=True,
        expected_tokens_per_sec=100,
    ),
    
    # Large Model (CPU offload)
    "meta-llama/Llama-3.1-70B-Instruct": VLLMModelConfig(
        model_id="meta-llama/Llama-3.1-70B-Instruct", 
        model_type="text",
        vram_required=24,
        ram_required=50,
        max_seq_len=4096,
        gpu_memory_utilization=0.95,
        expected_tokens_per_sec=25,
    ),
}

def get_optimal_model_config(model_id: str) -> VLLMModelConfig:
    """Get RTX 4090 optimized config"""
    if model_id in RTX_4090_MODELS:
        return RTX_4090_MODELS[model_id]
    
    # Default for unknown models
    return VLLMModelConfig(
        model_id=model_id,
        model_type="text",
        vram_required=16,
        ram_required=8,
        max_seq_len=4096,
        expected_tokens_per_sec=100,
    )

def get_available_vram() -> int:
    """Get VRAM in GB"""
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        return gpu_props.total_memory // (1024**3)
    return 0

def get_recommended_models() -> List[str]:
    """Models recommended for RTX 4090"""
    return [
        "microsoft/Phi-3-mini-4k-instruct",  # Fastest
        "meta-llama/Llama-3.1-8B-Instruct",  # Balanced
        "codellama/CodeLlama-13b-Instruct-hf",  # Code
        "Qwen/Qwen2-VL-7B-Instruct",  # Vision
    ]
```

**File: `src/llm_mcp/providers/vllm_v1/utils.py`**
```python
"""vLLM V1 Utilities"""

import torch
import psutil
from typing import Dict, Any
from .models import VLLMModelConfig

class VLLMHealthCheck:
    """System health for vLLM V1"""
    
    async def check_system(self) -> Dict[str, Any]:
        """Check RTX 4090 compatibility"""
        health = {
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_name": None,
            "vram_total": 0,
            "ram_total": 0,
            "v1_ready": self._check_v1_env(),
        }
        
        if health["cuda_available"]:
            gpu_props = torch.cuda.get_device_properties(0)
            health["gpu_name"] = gpu_props.name
            health["vram_total"] = gpu_props.total_memory // (1024**3)
        
        mem = psutil.virtual_memory()
        health["ram_total"] = mem.total // (1024**3)
        
        return health
    
    def _check_v1_env(self) -> bool:
        """Check V1 environment"""
        import os
        return os.getenv("VLLM_USE_V1") == "1"

def estimate_memory_usage(config: VLLMModelConfig) -> Dict[str, str]:
    """Memory usage estimate"""
    model_size = config.vram_required
    overhead = model_size * 0.2
    total = model_size + overhead
    
    return {
        "model_weights": f"{model_size}GB",
        "overhead": f"{overhead:.1f}GB",
        "total_estimate": f"{total:.1f}GB",
    }
```

## Step 2: Core Provider Implementation

**File: `src/llm_mcp/providers/vllm_v1/provider.py`**
```python
"""vLLM V1 Provider - RTX 4090 Optimized"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, AsyncGenerator, Any

from vllm.engine.arg_utils import AsyncEngineArgs  
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import SamplingParams

from ..base import BaseProvider, ProviderCapabilities
from .models import get_optimal_model_config, get_recommended_models
from .utils import VLLMHealthCheck, estimate_memory_usage

logger = logging.getLogger(__name__)

class VLLMv1Provider(BaseProvider):
    """vLLM V1 Provider for RTX 4090"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.engine: Optional[AsyncLLMEngine] = None
        self.current_model: Optional[str] = None
        self.model_config = None
        self.health_check = VLLMHealthCheck()
        
        # Enable V1 engine
        os.environ["VLLM_USE_V1"] = "1"
        os.environ["VLLM_ENABLE_PREFIX_CACHING"] = "1"
        
    async def initialize(self) -> bool:
        """Initialize with hardware check"""
        try:
            health = await self.health_check.check_system()
            
            if not health["cuda_available"]:
                logger.error("CUDA not available")
                return False
                
            logger.info(f"vLLM V1 ready - {health['gpu_name']}")
            return True
            
        except Exception as e:
            logger.error(f"Init failed: {e}")
            return False
    
    async def load_model(self, model_id: str, **kwargs) -> bool:
        """Load model with V1 optimizations"""
        try:
            self.model_config = get_optimal_model_config(model_id)
            logger.info(f"Loading {model_id}")
            
            # V1 Engine args
            engine_args = AsyncEngineArgs(
                model=model_id,
                tensor_parallel_size=self.model_config.tensor_parallel_size,
                gpu_memory_utilization=self.model_config.gpu_memory_utilization,
                max_seq_len=self.model_config.max_seq_len,
                enable_prefix_caching=True,
                enable_chunked_prefill=True,
                trust_remote_code=self.model_config.trust_remote_code,
                use_v2_block_manager=True,
            )
            
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.current_model = model_id
            
            expected_tps = self.model_config.expected_tokens_per_sec
            logger.info(f"✅ {model_id} loaded - Expected: {expected_tps} TPS")
            return True
            
        except Exception as e:
            logger.error(f"Load failed: {e}")
            return False
    
    async def generate_text(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate with V1 performance"""
        
        if not self.engine:
            raise RuntimeError("No model loaded")
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        
        request_id = f"req_{asyncio.get_event_loop().time()}"
        
        if stream:
            async for output in self.engine.generate(prompt, sampling_params, request_id):
                if output.outputs:
                    yield output.outputs[0].text
        else:
            final_text = ""
            async for output in self.engine.generate(prompt, sampling_params, request_id):
                if output.outputs:
                    final_text = output.outputs[0].text
            yield final_text
    
    async def list_available_models(self) -> List[str]:
        """RTX 4090 recommended models"""
        return get_recommended_models()
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Current model info"""
        if not self.current_model:
            return {}
            
        return {
            "model_id": self.current_model,
            "model_type": self.model_config.model_type if self.model_config else "text",
            "expected_tps": self.model_config.expected_tokens_per_sec if self.model_config else 0,
            "v1_engine": True,
            "flash_attention": True,
            "prefix_caching": True,
        }
    
    async def unload_model(self) -> bool:
        """Unload and cleanup"""
        try:
            if self.engine:
                del self.engine
                self.engine = None
                
            self.current_model = None
            self.model_config = None
            
            # GPU cleanup
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
            
            logger.info("Model unloaded")
            return True
            
        except Exception as e:
            logger.error(f"Unload failed: {e}")
            return False
    
    def get_capabilities(self) -> ProviderCapabilities:
        """Provider capabilities"""
        return ProviderCapabilities(
            supports_streaming=True,
            supports_vision=True,
            supports_function_calling=False,
            supports_json_mode=True,
            max_context_length=32768,
            supports_system_prompts=True,
        )
```

## Step 3: Provider Registration

**File: `src/llm_mcp/providers/__init__.py`** (Update)
```python
"""Provider Package with vLLM V1"""

from .base import BaseProvider, ProviderCapabilities
from .ollama.provider import OllamaProvider

# Import vLLM V1
try:
    from .vllm_v1.provider import VLLMv1Provider
    VLLM_V1_AVAILABLE = True
except ImportError as e:
    print(f"vLLM V1 not available: {e}")
    VLLMv1Provider = None
    VLLM_V1_AVAILABLE = False

# Provider registry
PROVIDERS = {
    "ollama": OllamaProvider,
}

if VLLM_V1_AVAILABLE:
    PROVIDERS["vllm_v1"] = VLLMv1Provider
    PROVIDERS["vllm"] = VLLMv1Provider  # Default vllm to V1

__all__ = [
    "BaseProvider",
    "ProviderCapabilities", 
    "OllamaProvider",
    "PROVIDERS",
    "VLLM_V1_AVAILABLE",
]

if VLLM_V1_AVAILABLE:
    __all__.append("VLLMv1Provider")
```

## Testing Commands

```bash
# Test vLLM V1 installation
python -c "
import os
os.environ['VLLM_USE_V1'] = '1'
from vllm.engine.async_llm_engine import AsyncLLMEngine
print('✅ vLLM V1 available')
"

# Test provider import
python -c "
from src.llm_mcp.providers.vllm_v1 import VLLMv1Provider
print('✅ Provider import successful')
"

# Test model config
python -c "
from src.llm_mcp.providers.vllm_v1.models import get_recommended_models
print('Recommended models:', get_recommended_models())
"
```

This gives you the core vLLM V1 provider implementation. Much cleaner and focused on your RTX 4090 setup. Ready for part 3 with MCP tools integration?
