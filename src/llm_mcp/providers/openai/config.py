"""OpenAI provider configuration."""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class OpenAIConfig(BaseModel):
    """Configuration for OpenAI provider."""
    
    api_key: Optional[str] = Field(None, description="OpenAI API key")
    base_url: str = Field("https://api.openai.com/v1", description="Base URL for OpenAI API")
    timeout: int = Field(30, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retries")
    default_model: str = Field("gpt-4o", description="Default model to use")
    
    # Model-specific settings
    max_tokens: int = Field(4096, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(1.0, description="Top-p sampling parameter")
    frequency_penalty: float = Field(0.0, description="Frequency penalty")
    presence_penalty: float = Field(0.0, description="Presence penalty")
    
    # Safety settings
    stop: Optional[list] = Field(None, description="Stop sequences")
    user: Optional[str] = Field(None, description="User identifier for tracking")
    
    class Config:
        env_prefix = "OPENAI_"
