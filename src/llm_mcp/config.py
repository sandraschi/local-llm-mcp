"""Configuration management for LLM MCP."""
import os
from typing import Dict, Any, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings."""
    
    # Server configuration
    host: str = Field("0.0.0.0", env="LLM_MCP_HOST")
    port: int = Field(8077, env="LLM_MCP_PORT")
    log_level: str = Field("info", env="LLM_MCP_LOG_LEVEL")
    debug: bool = Field(False, env="LLM_MCP_DEBUG")
    
    # Provider configurations
    providers: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "ollama": {
                "base_url": "http://localhost:11434",
                "timeout": 300,
                "enabled": True
            },
            # Add other provider defaults here
        }
    )
    
    # Model management
    default_provider: str = Field("ollama", env="LLM_MCP_DEFAULT_PROVIDER")
    default_model: str = Field("llama3", env="LLM_MCP_DEFAULT_MODEL")
    
    # Chat terminal settings
    chat_history_size: int = Field(100, env="LLM_MCP_CHAT_HISTORY_SIZE")
    chat_timeout: int = Field(300, env="LLM_MCP_CHAT_TIMEOUT")
    
    # File paths
    config_dir: str = Field(
        str(Path.home() / ".config" / "llm-mcp"),
        env="LLM_MCP_CONFIG_DIR"
    )
    data_dir: str = Field(
        str(Path.home() / ".local" / "share" / "llm-mcp"),
        env="LLM_MCP_DATA_DIR"
    )
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @validator("config_dir", "data_dir", pre=True)
    def create_dirs(cls, v: str) -> str:
        """Create directories if they don't exist."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    
    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["debug", "info", "warning", "error", "critical"]
        if v.lower() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.lower()


# Create a singleton instance of settings
settings = Settings()


def get_settings() -> Settings:
    """Get the application settings."""
    return settings


def update_settings(**kwargs) -> None:
    """Update application settings."""
    global settings
    settings = settings.copy(update=kwargs)


def get_provider_config(provider_name: str) -> Dict[str, Any]:
    """Get configuration for a specific provider.
    
    Args:
        provider_name: Name of the provider
        
    Returns:
        Provider configuration dictionary
    """
    provider_name = provider_name.lower()
    if provider_name not in settings.providers:
        raise ValueError(f"No configuration found for provider: {provider_name}")
    return settings.providers[provider_name]
