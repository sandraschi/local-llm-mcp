"""Configuration management for the LLM MCP Server."""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class ServerSettings(BaseModel):
    """Server configuration settings."""
    host: str = Field(default="0.0.0.0", description="Host to bind the server to")
    port: int = Field(default=8000, description="Port to run the server on")
    log_level: str = Field(default="info", description="Logging level")
    api_keys: List[str] = Field(
        default_factory=list,
        description="List of valid API keys (empty list means no auth required)"
    )

class ProviderSettings(BaseModel):
    """Provider-specific settings."""
    # Ollama settings
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for Ollama API"
    )
    
    # vLLM settings
    vllm_base_url: str = Field(
        default="http://localhost:8000",
        description="Base URL for vLLM API"
    )
    
    # LM Studio settings
    lmstudio_base_url: str = Field(
        default="http://localhost:1234",
        description="Base URL for LM Studio API"
    )
    
    # OpenAI settings
    openai_api_key: Optional[str] = Field(
        default=None,
        description="API key for OpenAI"
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for OpenAI API"
    )
    
    # Google Gemini settings
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="API key for Google Gemini"
    )
    
    # Perplexity AI settings
    perplexity_api_key: Optional[str] = Field(
        default=None,
        description="API key for Perplexity AI"
    )

class Settings(BaseSettings):
    """Application settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
    )
    
    # Server settings
    server: ServerSettings = Field(default_factory=ServerSettings)
    
    # Provider settings
    providers: ProviderSettings = Field(default_factory=ProviderSettings)
    
    @field_validator("server", mode='before')
    @classmethod
    def parse_server_settings(cls, v):
        """Parse server settings from environment variables."""
        if isinstance(v, dict):
            return v
        return {}
    
    @field_validator("providers", mode='before')
    @classmethod
    def parse_provider_settings(cls, v):
        """Parse provider settings from environment variables."""
        if isinstance(v, dict):
            return v
        return {}
    
    @classmethod
    def from_env(cls) -> 'Settings':
        """Create settings from environment variables."""
        return cls()

# Global settings instance
_settings = None

def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings
