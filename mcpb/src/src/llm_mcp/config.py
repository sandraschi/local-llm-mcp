"""Configuration management for LLM MCP server.

This module provides comprehensive configuration management with validation,
environment variable support, and runtime reconfiguration.
"""
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import yaml
from pydantic import BaseModel, Field, field_validator, ConfigDict
import structlog

logger = structlog.get_logger(__name__)

class LogLevel(str, Enum):
    """Supported log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class VLLMBackend(str, Enum):
    """vLLM attention backends."""
    FLASHINFER = "FLASHINFER"
    FLASHATTENTION = "FLASHATTENTION"
    XFORMERS = "XFORMERS"
    TORCH = "TORCH"

class VLLMConfig(BaseModel):
    """vLLM 1.0+ configuration with all modern features."""
    
    # Engine settings
    use_v1_engine: bool = True
    attention_backend: VLLMBackend = VLLMBackend.FLASHINFER
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    
    # Memory management
    gpu_memory_utilization: float = Field(default=0.9, ge=0.1, le=1.0)
    swap_space: str = "4GB"
    cpu_offload: bool = False
    
    # Parallelism
    tensor_parallel_size: int = Field(default=1, ge=1, le=8)
    pipeline_parallel_size: int = Field(default=1, ge=1, le=8)
    
    # Performance tuning
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: int = 256
    max_model_len: Optional[int] = None
    
    # Multimodal support (vLLM 1.0+ feature)
    enable_vision: bool = True
    enable_audio: bool = False  # Future feature
    max_image_input_size: int = 2048
    
    # Advanced features
    enable_lora: bool = True
    enable_tool_calling: bool = True
    enable_structured_output: bool = True
    
    # Quantization
    quantization: Optional[str] = None  # "awq", "gptq", "fp8", etc.
    load_format: str = "auto"
    dtype: str = "auto"

class ServerConfig(BaseModel):
    """Server configuration."""
    name: str = "Local LLM MCP Server"
    version: str = "1.0.0"
    host: str = "localhost"
    port: int = 8000
    log_level: LogLevel = LogLevel.INFO
    log_file: Path = Path("logs/llm_mcp.log")
    log_rotation_size: str = "10MB"
    log_retention_count: int = 5

class ModelConfig(BaseModel):
    """Model configuration."""
    default_provider: str = "vllm"
    available_providers: List[str] = ["vllm", "ollama", "openai", "anthropic"]
    
    # Model paths and settings
    model_cache_dir: Path = Path("models")
    default_model: str = "microsoft/Phi-3.5-mini-instruct"
    
    # Generation defaults
    default_max_tokens: int = 2048
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    
    @field_validator('model_cache_dir')
    @classmethod
    def validate_model_cache_dir(cls, v):
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

class Config(BaseModel):
    """Main configuration class."""
    server: ServerConfig = ServerConfig()
    model: ModelConfig = ModelConfig()
    vllm: VLLMConfig = VLLMConfig()
    
    # Runtime configuration
    config_path: Optional[Path] = None
    environment_overrides: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        env_prefix="LLM_MCP_",
        env_nested_delimiter="__"
    )
    
    @classmethod
    def load(cls, config_path: Optional[Union[str, Path]] = None) -> "Config":
        """Load configuration from file and environment variables."""
        # Default config path
        if config_path is None:
            config_path = Path("config.yaml")
        else:
            config_path = Path(config_path)
        
        # Load base configuration
        config_data = {}
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
                logger.info("Configuration loaded from file", path=str(config_path))
            except Exception as e:
                logger.warning("Failed to load config file", path=str(config_path), error=str(e))
        else:
            logger.info("No config file found, using defaults", path=str(config_path))
        
        # Apply environment variable overrides
        env_overrides = cls._load_env_overrides()
        if env_overrides:
            logger.info("Applied environment variable overrides", count=len(env_overrides))
            config_data = cls._merge_config(config_data, env_overrides)
        
        # Create configuration instance
        config = cls(**config_data)
        config.config_path = config_path
        config.environment_overrides = env_overrides
        
        # Ensure directories exist
        config.server.log_file.parent.mkdir(parents=True, exist_ok=True)
        config.model.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        return config
    
    @classmethod
    def _load_env_overrides(cls) -> Dict[str, Any]:
        """Load configuration overrides from environment variables."""
        overrides = {}
        prefix = "LLM_MCP_"
        
        # Map environment variables to config structure
        env_mapping = {
            f"{prefix}LOG_LEVEL": ("server", "log_level"),
            f"{prefix}HOST": ("server", "host"),
            f"{prefix}PORT": ("server", "port"),
            f"{prefix}DEFAULT_PROVIDER": ("model", "default_provider"),
            f"{prefix}DEFAULT_MODEL": ("model", "default_model"),
            f"{prefix}VLLM_GPU_MEMORY": ("vllm", "gpu_memory_utilization"),
            f"{prefix}VLLM_TENSOR_PARALLEL": ("vllm", "tensor_parallel_size"),
            f"{prefix}VLLM_ENABLE_VISION": ("vllm", "enable_vision"),
            f"{prefix}VLLM_ATTENTION_BACKEND": ("vllm", "attention_backend"),
        }
        
        for env_var, (section, key) in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                # Type conversion
                if key in ["port", "tensor_parallel_size"]:
                    value = int(value)
                elif key == "gpu_memory_utilization":
                    value = float(value)
                elif key in ["enable_vision"]:
                    value = value.lower() in ("true", "1", "yes", "on")
                
                if section not in overrides:
                    overrides[section] = {}
                overrides[section][key] = value
        
        return overrides
    
    @classmethod
    def _merge_config(cls, base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration dictionaries."""
        result = base.copy()
        for key, value in overrides.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = cls._merge_config(result[key], value)
            else:
                result[key] = value
        return result
    
    def save(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Save configuration to file."""
        if config_path is None:
            config_path = self.config_path or Path("config.yaml")
        else:
            config_path = Path(config_path)
        
        # Convert to dictionary and remove runtime fields
        config_dict = self.dict(exclude={"config_path", "environment_overrides"})
        
        # Convert Path objects to strings for YAML serialization
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj
        
        config_dict = convert_paths(config_dict)
        
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            logger.info("Configuration saved", path=str(config_path))
        except Exception as e:
            logger.error("Failed to save configuration", path=str(config_path), error=str(e))
            raise
    
    def update_runtime(self, **kwargs) -> None:
        """Update configuration at runtime."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info("Configuration updated", key=key, value=value)
            else:
                logger.warning("Unknown configuration key", key=key)
    
    def get_vllm_env_vars(self) -> Dict[str, str]:
        """Get vLLM environment variables for process configuration."""
        env_vars = {}
        
        if self.vllm.use_v1_engine:
            env_vars["VLLM_USE_V1"] = "1"
        
        if self.vllm.enable_prefix_caching:
            env_vars["VLLM_ENABLE_PREFIX_CACHING"] = "1"
        
        if self.vllm.attention_backend:
            env_vars["VLLM_ATTENTION_BACKEND"] = self.vllm.attention_backend.value
        
        env_vars["VLLM_GPU_MEMORY_UTILIZATION"] = str(self.vllm.gpu_memory_utilization)
        
        return env_vars
    
    def validate_hardware_compatibility(self) -> Dict[str, Any]:
        """Validate hardware compatibility and suggest optimizations."""
        import torch
        import psutil
        
        compatibility = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "total_memory": psutil.virtual_memory().total,
            "recommendations": []
        }
        
        # GPU recommendations
        if compatibility["cuda_available"]:
            for i in range(compatibility["cuda_device_count"]):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory
                compatibility[f"gpu_{i}_memory"] = gpu_memory
                
                # Memory utilization recommendations
                if self.vllm.gpu_memory_utilization > 0.95:
                    compatibility["recommendations"].append(
                        f"GPU {i}: Consider reducing gpu_memory_utilization below 0.95 to avoid OOM errors"
                    )
        else:
            compatibility["recommendations"].append("No CUDA GPUs detected - vLLM will run in CPU mode (very slow)")
        
        # Tensor parallelism recommendations
        if self.vllm.tensor_parallel_size > compatibility["cuda_device_count"]:
            compatibility["recommendations"].append(
                f"tensor_parallel_size ({self.vllm.tensor_parallel_size}) exceeds available GPUs ({compatibility['cuda_device_count']})"
            )
        
        return compatibility

# Global configuration instance
_config_instance: Optional[Config] = None

def get_config() -> Config:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config.load()
    return _config_instance

def reload_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Reload the global configuration."""
    global _config_instance
    _config_instance = Config.load(config_path)
    return _config_instance
