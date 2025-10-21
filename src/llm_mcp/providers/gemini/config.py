"""Gemini provider configuration."""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

class GeminiConfig(BaseModel):
    """Configuration for Gemini provider."""
    
    api_key: Optional[str] = Field(None, description="Google AI API key")
    base_url: str = Field("https://generativelanguage.googleapis.com/v1beta", description="Base URL for Gemini API")
    timeout: int = Field(30, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retries")
    default_model: str = Field("gemini-1.5-pro", description="Default model to use")
    
    # Model-specific settings
    max_tokens: int = Field(8192, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.95, description="Top-p sampling parameter")
    top_k: int = Field(40, description="Top-k sampling parameter")
    
    # Safety settings
    stop_sequences: Optional[list] = Field(None, description="Stop sequences")
    
    model_config = ConfigDict(
        env_prefix="GEMINI_"
    )
