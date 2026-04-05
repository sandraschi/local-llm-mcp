"""Perplexity provider configuration."""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

class PerplexityConfig(BaseModel):
    """Configuration for Perplexity provider."""
    
    api_key: Optional[str] = Field(None, description="Perplexity API key")
    base_url: str = Field("https://api.perplexity.ai", description="Base URL for Perplexity API")
    timeout: int = Field(30, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retries")
    default_model: str = Field("llama-3.1-sonar-small-128k-online", description="Default model to use")
    
    # Model-specific settings
    max_tokens: int = Field(4096, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Top-p sampling parameter")
    top_k: int = Field(40, description="Top-k sampling parameter")
    
    # Safety settings
    stop: Optional[list] = Field(None, description="Stop sequences")
    
    model_config = ConfigDict(
        env_prefix="PERPLEXITY_"
    )
