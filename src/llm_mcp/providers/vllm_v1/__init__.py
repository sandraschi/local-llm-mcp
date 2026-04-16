"""vLLM V1 provider package."""

from .config import VLLMv1Config
from .models import ModelCapability, ModelType, VLLMModel, VLLMMultimodalModel
from .provider import VLLMv1Provider

__all__ = ["ModelCapability", "ModelType", "VLLMModel", "VLLMMultimodalModel", "VLLMv1Config", "VLLMv1Provider"]

# Current vLLM version we're targeting
VLLM_VERSION = "0.8.3"  # Latest version compatible with Python 3.13
VLLM_MINIMUM_VERSION = "0.8.1"  # Minimum for V1 engine support

# Key features available in v0.10+
FEATURES = {
    "v1_engine": True,
    "flashattention_3": True,
    "prefix_caching": True,
    "multimodal_support": True,
    "distributed_inference": True,
    "structured_outputs": True,
    "tool_calling": True,
    "zero_config_optimization": True,
}
