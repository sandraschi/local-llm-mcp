"""Anthropic provider configuration."""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

class AnthropicConfig(BaseModel):
    """Configuration for Anthropic provider."""
    
    api_key: Optional[str] = Field(None, description="Anthropic API key")
    base_url: str = Field("https://api.anthropic.com", description="Base URL for Anthropic API")
    timeout: int = Field(30, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retries")
    default_model: str = Field("claude-3-sonnet-20240229", description="Default model to use")
    
    # Model-specific settings
    max_tokens: int = Field(4096, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(1.0, description="Top-p sampling parameter")
    top_k: int = Field(40, description="Top-k sampling parameter")
    
    # Safety settings
    stop_sequences: Optional[list] = Field(None, description="Stop sequences")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata to include in requests")
    
    model_config = ConfigDict(
        env_prefix="ANTHROPIC_"
    )
