"""Configuration models for Hugging Face provider."""

from pydantic import Field, HttpUrl, model_validator
from typing import Optional, List, Dict, Any

class HuggingFaceConfig:
    """Configuration for Hugging Face provider.
    
    Attributes:
        api_key: Hugging Face API token (required for private/gated models)
        model_name: Name of the model to use (e.g., 'gpt2', 'bigscience/bloom')
        model_revision: Specific model version or branch to use
        device: Device to run the model on ('cuda', 'cpu', 'auto')
        trust_remote_code: Whether to trust remote code in the model
        use_auth_token: Alias for api_key, maintained for backward compatibility
        task: Task to use for the model (e.g., 'text-generation')
        model_kwargs: Additional keyword arguments to pass to from_pretrained()
        pipeline_kwargs: Additional keyword arguments to pass to pipeline()
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt2",
        model_revision: Optional[str] = None,
        device: str = "auto",
        trust_remote_code: bool = False,
        use_auth_token: Optional[str] = None,
        task: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        pipeline_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize Hugging Face configuration."""
        self.api_key = api_key or use_auth_token
        self.model_name = model_name
        self.model_revision = model_revision
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.task = task or self._infer_task(model_name)
        self.model_kwargs = model_kwargs or {}
        self.pipeline_kwargs = pipeline_kwargs or {}
    
    @staticmethod
    def _infer_task(model_name: str) -> str:
        """Infer the task from the model name if not specified."""
        model_name_lower = model_name.lower()
        if any(t in model_name_lower for t in ["gpt", "llama", "bloom", "opt"]):
            return "text-generation"
        elif any(t in model_name_lower for t in ["t5", "bart", "pegasus"]):
            return "text2text-generation"
        elif any(t in model_name_lower for t in ["bert", "roberta", "distilbert"]):
            return "fill-mask"
        return "text-generation"  # Default to text generation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return {
            "api_key": self.api_key,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "device": self.device,
            "trust_remote_code": self.trust_remote_code,
            "task": self.task,
            "model_kwargs": self.model_kwargs,
            "pipeline_kwargs": self.pipeline_kwargs,
        }
