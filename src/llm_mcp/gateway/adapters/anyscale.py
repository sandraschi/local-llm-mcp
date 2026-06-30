"""Anyscale adapter — OpenAI-compatible."""

from llm_mcp.gateway.adapters.openai import OpenAIAdapter
from llm_mcp.gateway.base import register_provider


@register_provider("anyscale")
class AnyscaleAdapter(OpenAIAdapter):
    provider = "anyscale"
    base_url = "https://api.endpoints.anyscale.com/v1"
    api_key_env = "ANYSCALE_API_KEY"
