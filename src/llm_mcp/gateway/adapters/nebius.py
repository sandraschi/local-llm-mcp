"""Nebius AI adapter — OpenAI-compatible."""

from llm_mcp.gateway.adapters.openai import OpenAIAdapter
from llm_mcp.gateway.base import register_provider


@register_provider("nebius")
class NebiusAdapter(OpenAIAdapter):
    provider = "nebius"
    base_url = "https://api.nebius.ai/v1"
    api_key_env = "NEBIUS_API_KEY"
