"""Configuration for vLLM V1 provider with v1.0.0+ support."""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List, Literal, Dict, Any, Union
from pathlib import Path

class VLLMv1Config(BaseModel):
    """Configuration for vLLM V1 provider with v1.0.0+ support.
    
    This configuration includes all major parameters from vLLM 1.0.0+ for optimal
    performance and flexibility.
    """
    
    # ===== Server Configuration =====
    host: str = Field(
        "0.0.0.0",
        description="Host to bind the vLLM server to"
    )
    port: int = Field(
        8001,
        ge=1024,
        le=65535,
        description="Port to run the vLLM server on"
    )
    base_url: str = Field(
        "http://localhost:8001",
        description="Base URL for vLLM API endpoints"
    )
    
    # ===== Model Loading =====
    model: str = Field(
        ...,
        description="Name or path of the model to load"
    )
    tokenizer: Optional[str] = Field(
        None,
        description="Name or path of the tokenizer (defaults to model)"
    )
    tokenizer_mode: Literal["auto", "slow"] = Field(
        "auto",
        description="Tokenizer mode (auto uses fast tokenizer if available, otherwise slow)"
    )
    trust_remote_code: bool = Field(
        False,
        description="Trust remote code when loading the model"
    )
    download_dir: Optional[Union[str, Path]] = Field(
        None,
        description="Directory to download and load the model"
    )
    
    # ===== Engine Configuration =====
    tensor_parallel_size: int = Field(
        1,
        ge=1,
        description="Number of GPUs to use for distributed inference"
    )
    max_parallel_loading_workers: Optional[int] = Field(
        None,
        description="Maximum number of workers to use for model loading"
    )
    block_size: int = Field(
        16,
        description="Size of a block in number of tokens"
    )
    use_v2_block_manager: bool = Field(
        True,
        description="Use the v2 block manager for better memory management"
    )
    
    # ===== KV Cache Configuration =====
    gpu_memory_utilization: float = Field(
        0.9,
        ge=0.0,
        le=1.0,
        description="Fraction of GPU memory to use for the model"
    )
    swap_space: int = Field(
        4,
        ge=0,
        description="CPU swap space size (GiB per GPU) for KV cache"
    )
    
    # ===== Attention Configuration =====
    max_seq_len: int = Field(
        8192,
        ge=1,
        description="Maximum sequence length"
    )
    max_num_batched_tokens: Optional[int] = Field(
        None,
        description="Maximum number of batched tokens per iteration"
    )
    max_num_seqs: int = Field(
        256,
        ge=1,
        description="Maximum number of sequences per batch"
    )
    
    # ===== Performance Optimization =====
    enable_chunked_prefill: bool = Field(
        True,
        description="Enable chunked prefill for long sequences"
    )
    preemption_mode: Literal["recompute", "swap"] = Field(
        "recompute",
        description="Preemption mode for long sequences"
    )
    
    # ===== Quantization =====
    quantization: Optional[Literal["awq", "gptq", "squeezellm"]] = Field(
        None,
        description="Quantization method to use"
    )
    
    # ===== Logging and Monitoring =====
    log_requests: bool = Field(
        False,
        description="Log all requests to the server"
    )
    log_stats: bool = Field(
        False,
        description="Log performance statistics"
    )
    
    # ===== Advanced Settings =====
    max_model_len: Optional[int] = Field(
        None,
        description="Maximum model length (overrides model config)"
    )
    seed: int = Field(
        42,
        description="Random seed for reproducibility"
    )
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_base_url(cls, v, info):
        """Ensure base_url is properly formatted."""
        values = info.data
        if 'host' in values and 'port' in values:
            return f"http://{values['host']}:{values['port']}"
        return v
    # Distributed inference
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    
    # Auto-optimization (V1 features)
    enable_chunked_prefill: bool = True
    num_scheduler_steps: int = 1  # Auto-tuned in V1
    
    # Model loading
    model_download_dir: Optional[str] = None
    trust_remote_code: bool = False
    
    # Logging
    log_level: str = "INFO"
    disable_log_requests: bool = False
    
    model_config = ConfigDict(
        env_prefix="VLLM_"
    )
