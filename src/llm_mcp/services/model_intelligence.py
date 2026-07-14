"""Service for providing rich intelligence and metadata about LLMs."""

import logging
from typing import Any

from llm_mcp.api.v1.models.llm import ModelIntelligence

logger = logging.getLogger(__name__)

# THE 2026 VANGUARD REGISTRY
# Curated metadata for the elite 2025/2026 SOTA models
VANGUARD_REGISTRY: dict[str, dict[str, Any]] = {
    "gemma-4-7b": {
        "hf_id": "google/gemma-4-7b-it",
        "developer": "Google",
        "release_date": "2026-02-15",
        "strengths": ["Extreme logical reasoning", "Native multimodal (video/audio)", "Low latency"],
        "weaknesses": ["Sensitive to system prompt safety filters", "Occasional brevity"],
        "best_for": "Fast, high-quality reasoning and vision tasks.",
        "vram_required_gb": 6.0,
        "quantization_info": "Highly optimized for GGUF-Q4_K_M and EXL2-4.0bpw.",
        "model_card_url": "https://huggingface.co/google/gemma-4-7b-it",
    },
    "gemma-4-27b": {
        "hf_id": "google/gemma-4-27b-it",
        "developer": "Google",
        "release_date": "2026-03-01",
        "strengths": ["Near-GPT-5 performance", "1M+ context window", "Expert coding assistant"],
        "weaknesses": ["VRAM intensive (Needs 24GB+ for full context)", "Slower than 7B"],
        "best_for": "Complex coding, long-document analysis, and deep reasoning.",
        "vram_required_gb": 18.0,
        "quantization_info": "Recommended: EXL2-6.0bpw for dual 3090/4090 setups.",
        "model_card_url": "https://huggingface.co/google/gemma-4-27b-it",
    },
    "llama-4-8b": {
        "hf_id": "meta-llama/Llama-4-8B-Instruct",
        "developer": "Meta",
        "release_date": "2025-11-20",
        "strengths": ["Best-in-class open small model", "Very instruction-following", "Huge ecosystem support"],
        "weaknesses": ["Context window limited compared to Gemma 4", "Less natively multimodal"],
        "best_for": "General purpose chat and tool-calling.",
        "vram_required_gb": 7.0,
        "quantization_info": "Flawless performance at Q8_0 or EXL2-8.0bpw.",
        "model_card_url": "https://huggingface.co/meta-llama/Llama-4-8B-Instruct",
    },
    "llama-4-70b": {
        "hf_id": "meta-llama/Llama-4-70B-Instruct",
        "developer": "Meta",
        "release_date": "2026-01-10",
        "strengths": ["Industry benchmark for open weights", "Superior nuance", "Agentic planning"],
        "weaknesses": ["Highly compute intensive", "Requires multi-GPU for inference"],
        "best_for": "Enterprise-grade orchestration and complex task decomposition.",
        "vram_required_gb": 45.0,
        "quantization_info": "Use Q4_K_M for 48GB VRAM (Dual GPU).",
        "model_card_url": "https://huggingface.co/meta-llama/Llama-4-70B-Instruct",
    },
    "claude-4-sonnet": {
        "hf_id": "anthropic/claude-4-sonnet",
        "developer": "Anthropic",
        "release_date": "2026-02-10",
        "strengths": ["SOTA Artifact generation", "Unmatched tone compliance", "Perfect reasoning"],
        "weaknesses": ["Closed weights (OpenRouter only)", "Rate limits"],
        "best_for": "Creative writing, complex artifacts, and high-precision tasks.",
        "vram_required_gb": 0.0,  # Cloud based
        "model_card_url": "https://www.anthropic.com/claude",
    },
    "deepseek-v4": {
        "hf_id": "deepseek-ai/DeepSeek-V4",
        "developer": "DeepSeek",
        "release_date": "2026-03-15",
        "strengths": ["Unbeatable price/performance", "MoE efficiency", "Coding specialist"],
        "weaknesses": ["Can be repetitive in creative tasks", "Closed ecosystem initially"],
        "best_for": "Cheap, fast, high-quality coding and math.",
        "vram_required_gb": 40.0,
        "model_card_url": "https://huggingface.co/deepseek-ai/DeepSeek-V4",
    },
}


class ModelIntelligenceService:
    """Service to provide rich metadata for LLMs."""

    def __init__(self):
        self.registry = VANGUARD_REGISTRY

    def get_intelligence(self, model_id: str) -> ModelIntelligence | None:
        """Get rich metadata for a model by its ID.

        Args:
            model_id: The identifier for the model.

        Returns:
            ModelIntelligence object if found, else enriched dynamic metadata.
        """
        # Normalize model_id (lower case and strip common provider prefixes)
        normalized_id = model_id.lower()
        if "/" in normalized_id:
            normalized_id = normalized_id.split("/")[-1]

        # Check carefully for fuzzy matches in the registry
        registry_data = None
        for key in self.registry:
            if key in normalized_id or normalized_id in key:
                registry_data = self.registry[key]
                break

        if registry_data:
            return ModelIntelligence(**registry_data)

        # Fallback to dynamic metadata detection
        return self._detect_metadata(model_id)

    def _detect_metadata(self, model_id: str) -> ModelIntelligence:
        """Infer metadata for models not in the curated registry."""
        model_id_lower = model_id.lower()

        # Determine if legacy (pre-2025)
        is_legacy = any(x in model_id_lower for x in ["llama-3", "gemma-2", "gpt-4-turbo", "claude-3"])

        # Estimate VRAM
        vram_req = self.estimate_vram_requirement(model_id)

        developer = "Unknown"
        if "meta" in model_id_lower or "llama" in model_id_lower:
            developer = "Meta"
        elif "google" in model_id_lower or "gemma" in model_id_lower:
            developer = "Google"
        elif "anthropic" in model_id_lower or "claude" in model_id_lower:
            developer = "Anthropic"
        elif "openai" in model_id_lower or "gpt" in model_id_lower:
            developer = "OpenAI"
        elif "mistral" in model_id_lower:
            developer = "Mistral"
        elif "microsoft" in model_id_lower or "phi" in model_id_lower:
            developer = "Microsoft"

        return ModelIntelligence(
            hf_id=model_id if "/" in model_id else None,
            developer=developer,
            is_legacy=is_legacy,
            vram_required_gb=vram_req,
            best_for="General purpose interaction." if not is_legacy else "Legacy support and baseline testing.",
            strengths=["Standard LLM capabilities"] if not is_legacy else ["High compatibility", "Stable performance"],
            weaknesses=["Metadata not yet curated for this specific model."]
            if not is_legacy
            else ["Outdated context/knowledge base (Pre-2025)"],
        )

    def estimate_vram_requirement(self, model_id: str) -> float:
        """Estimate VRAM requirement in GB based on model identifier strings."""
        id_lower = model_id.lower()

        # Tiny/Special cases
        if any(x in id_lower for x in ["tinyllama", "phi-2", "phi-3-mini", "stable-lm-2"]):
            return 3.0
        if "phi-3-medium" in id_lower or "phi-4" in id_lower:
            return 10.0

        # Parameter count patterns (e.g. 7B, 13B, 70B)
        import re

        match = re.search(r"(\d+)[bm]", id_lower)
        if match:
            count = int(match.group(1))
            unit = match.group(0)[-1]

            if unit == "b":
                # 4-bit quantization baseline: ~0.6-0.8 GB per Billion params + overhead
                if count <= 3:
                    return 4.0
                if count <= 8:
                    return 6.0
                if count <= 14:
                    return 12.0
                if count <= 30:
                    return 20.0
                if count <= 75:
                    return 45.0
                return count * 0.7
            if unit == "m":
                return 2.0

        # Default fallback
        return 8.0

    def get_compatibility(self, model_vram: float, gpu_vram_gb: float) -> str:
        """Compare model requirement against available hardware.

        Returns one of: 'READY', 'TIGHT', 'OOM', 'UNKNOWN'
        """
        if gpu_vram_gb <= 0:
            return "UNKNOWN"

        if model_vram == 0:  # Cloud model
            return "READY"

        if gpu_vram_gb >= (model_vram + 3.0):  # Healthy buffer
            return "READY"

        if gpu_vram_gb >= model_vram:  # Fits but tight
            return "TIGHT"

        return "OOM"
