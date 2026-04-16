"""OpenRouter provider configuration."""


from pydantic import Field

from ...config.base import BaseProviderConfig


class OpenRouterConfig(BaseProviderConfig):
    """Configuration for the OpenRouter provider."""

    api_key: str | None = Field(
        None,
        description="OpenRouter API key",
        env="OPENROUTER_API_KEY"
    )

    base_url: str = Field(
        "https://openrouter.ai/api/v1",
        description="OpenRouter API base URL"
    )

    default_model: str = Field(
        "google/gemma-2-9b-it",
        description="Default model to use for generation"
    )

    timeout: int = Field(
        60,
        description="Request timeout in seconds"
    )

    max_retries: int = Field(
        2,
        description="Maximum number of retries for failed requests"
    )

    # OpenRouter specific headers
    site_url: str | None = Field(
        None,
        description="Site URL for OpenRouter rankings"
    )

    site_name: str | None = Field(
        "Local LLM MCP",
        description="Site name for OpenRouter rankings"
    )
