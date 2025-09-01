"""Model definitions for vLLM V1 provider."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class ModelType(str, Enum):
    """Supported model types."""
    TEXT = "text"
    MULTIMODAL = "multimodal" 
    VISION = "vision"
    AUDIO = "audio"
    CODE = "code"
    EMBEDDING = "embedding"


class ModelCapability(str, Enum):
    """Model capabilities."""
    TEXT_GENERATION = "text-generation"
    CHAT = "chat"
    VISION = "vision"
    AUDIO_UNDERSTANDING = "audio-understanding"
    CODE_GENERATION = "code-generation"
    EMBEDDING = "embedding"
    TOOL_CALLING = "tool-calling"


class VLLMModel(BaseModel):
    """Base vLLM model definition."""
    
    id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Human-readable model name")
    type: ModelType = Field(..., description="Model type")
    capabilities: List[ModelCapability] = Field(default_factory=list)
    
    # Resource requirements
    vram_required: str = Field(..., description="VRAM requirement (e.g., '8GB')")
    context_length: int = Field(default=4096, description="Maximum context length")
    
    # Performance optimization
    tensor_parallel_recommended: Optional[int] = Field(None, description="Recommended tensor parallelism")
    pipeline_parallel_recommended: Optional[int] = Field(None, description="Recommended pipeline parallelism")
    
    # Configuration
    supports_v1_engine: bool = Field(default=True, description="Supports vLLM V1 engine")
    supports_prefix_caching: bool = Field(default=True, description="Supports prefix caching")
    supports_flashattention: bool = Field(default=True, description="Supports FlashAttention")
    
    class Config:
        use_enum_values = True


class VLLMMultimodalModel(VLLMModel):
    """Multimodal model with vision/audio capabilities."""
    
    # Multimodal specific
    max_images: Optional[int] = Field(None, description="Maximum images per request")
    max_video_frames: Optional[int] = Field(None, description="Maximum video frames")
    max_audio_length: Optional[int] = Field(None, description="Maximum audio length (seconds)")
    
    # Vision settings
    max_image_size: int = Field(default=2048, description="Maximum image resolution")
    vision_embedding_dim: Optional[int] = Field(None, description="Vision embedding dimension")


# Pre-defined model configurations for vLLM v0.10+
VLLM_MODELS = {
    # Llama 3.1/3.2 family
    "meta-llama/Llama-3.1-8B-Instruct": VLLMModel(
        id="meta-llama/Llama-3.1-8B-Instruct",
        name="Llama 3.1 8B Instruct",
        type=ModelType.TEXT,
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT, ModelCapability.TOOL_CALLING],
        vram_required="8GB",
        context_length=128000,
        supports_v1_engine=True
    ),
    
    "meta-llama/Llama-3.1-70B-Instruct": VLLMModel(
        id="meta-llama/Llama-3.1-70B-Instruct", 
        name="Llama 3.1 70B Instruct",
        type=ModelType.TEXT,
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT, ModelCapability.TOOL_CALLING],
        vram_required="40GB",
        context_length=128000,
        tensor_parallel_recommended=4,
        supports_v1_engine=True
    ),

    "meta-llama/Llama-3.2-11B-Vision-Instruct": VLLMMultimodalModel(
        id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        name="Llama 3.2 11B Vision Instruct", 
        type=ModelType.MULTIMODAL,
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT, ModelCapability.VISION],
        vram_required="12GB",
        context_length=128000,
        max_images=20,
        max_image_size=2048,
        supports_v1_engine=True
    ),
    
    # Qwen2-VL (Excellent multimodal performance)
    "Qwen/Qwen2-VL-7B-Instruct": VLLMMultimodalModel(
        id="Qwen/Qwen2-VL-7B-Instruct",
        name="Qwen2-VL 7B Instruct",
        type=ModelType.MULTIMODAL,
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT, ModelCapability.VISION],
        vram_required="12GB",
        context_length=32768,
        max_images=20,
        max_video_frames=32,
        supports_v1_engine=True
    ),
    
    "Qwen/Qwen2-VL-72B-Instruct": VLLMMultimodalModel(
        id="Qwen/Qwen2-VL-72B-Instruct",
        name="Qwen2-VL 72B Instruct",
        type=ModelType.MULTIMODAL,
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT, ModelCapability.VISION],
        vram_required="80GB",
        context_length=32768,
        max_images=20,
        tensor_parallel_recommended=8,
        supports_v1_engine=True
    ),
    
    # LLaVA family (Popular vision models)
    "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf": VLLMMultimodalModel(
        id="llava-hf/llava-onevision-qwen2-7b-ov-chat-hf",
        name="LLaVA OneVision 7B",
        type=ModelType.MULTIMODAL,
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT, ModelCapability.VISION],
        vram_required="10GB",
        context_length=4096,
        max_images=10,
        supports_v1_engine=True
    ),
    
    # Code models
    "deepseek-ai/deepseek-coder-33b-instruct": VLLMModel(
        id="deepseek-ai/deepseek-coder-33b-instruct",
        name="DeepSeek Coder 33B Instruct",
        type=ModelType.CODE,
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT, ModelCapability.CODE_GENERATION],
        vram_required="20GB",
        context_length=16384,
        tensor_parallel_recommended=2,
        supports_v1_engine=True
    ),
    
    "microsoft/DialoGPT-medium": VLLMModel(
        id="microsoft/DialoGPT-medium", 
        name="DialoGPT Medium",
        type=ModelType.TEXT,
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT],
        vram_required="2GB",
        context_length=1024,
        supports_v1_engine=True
    ),
    
    # Phi-4 (Microsoft's latest)
    "microsoft/Phi-4": VLLMModel(
        id="microsoft/Phi-4",
        name="Phi-4 14B",
        type=ModelType.TEXT,
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT, ModelCapability.CODE_GENERATION],
        vram_required="14GB",
        context_length=16384,
        supports_v1_engine=True
    ),
    
    # Gemma 2 family
    "google/gemma-2-9b-it": VLLMModel(
        id="google/gemma-2-9b-it",
        name="Gemma 2 9B Instruct",
        type=ModelType.TEXT,
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT],
        vram_required="9GB",
        context_length=8192,
        supports_v1_engine=True
    ),
    
    # Mistral family
    "mistralai/Mistral-7B-Instruct-v0.3": VLLMModel(
        id="mistralai/Mistral-7B-Instruct-v0.3",
        name="Mistral 7B Instruct v0.3",
        type=ModelType.TEXT,
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT, ModelCapability.TOOL_CALLING],
        vram_required="7GB",
        context_length=32768,
        supports_v1_engine=True
    ),
    
    "mistralai/Mixtral-8x7B-Instruct-v0.1": VLLMModel(
        id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        name="Mixtral 8x7B Instruct",
        type=ModelType.TEXT,
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT, ModelCapability.TOOL_CALLING],
        vram_required="24GB",
        context_length=32768,
        tensor_parallel_recommended=2,
        supports_v1_engine=True
    ),
}


def get_model_by_id(model_id: str) -> Optional[VLLMModel]:
    """Get model configuration by ID."""
    return VLLM_MODELS.get(model_id)


def get_models_by_type(model_type: ModelType) -> List[VLLMModel]:
    """Get all models of a specific type."""
    return [model for model in VLLM_MODELS.values() if model.type == model_type]


def get_multimodal_models() -> List[VLLMMultimodalModel]:
    """Get all multimodal models."""
    return [model for model in VLLM_MODELS.values() 
            if isinstance(model, VLLMMultimodalModel)]


def estimate_vram_usage(model_id: str, tensor_parallel: int = 1) -> str:
    """Estimate VRAM usage for distributed setup."""
    model = get_model_by_id(model_id)
    if not model:
        return "Unknown"
    
    # Simple estimation: divide by tensor parallel size
    vram_gb = int(''.join(filter(str.isdigit, model.vram_required)))
    distributed_vram = max(1, vram_gb // tensor_parallel)
    return f"{distributed_vram}GB per GPU"


def get_recommended_parallelism(model_id: str, available_gpus: int = 1) -> Dict[str, int]:
    """Get recommended parallelism settings."""
    model = get_model_by_id(model_id)
    if not model:
        return {"tensor_parallel": 1, "pipeline_parallel": 1}
    
    # Use recommended settings if available, otherwise estimate
    tensor_parallel = min(
        model.tensor_parallel_recommended or 1,
        available_gpus
    )
    
    pipeline_parallel = min(
        model.pipeline_parallel_recommended or 1,
        available_gpus // tensor_parallel
    )
    
    return {
        "tensor_parallel": tensor_parallel,
        "pipeline_parallel": pipeline_parallel
    }
