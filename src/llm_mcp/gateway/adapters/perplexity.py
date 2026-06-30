"""Perplexity adapter — OpenAI-compatible."""

from llm_mcp.gateway.adapters.openai import OpenAIAdapter
from llm_mcp.gateway.base import register_provider


@register_provider("perplexity")
class PerplexityAdapter(OpenAIAdapter):
    provider = "perplexity"
    base_url = "https://api.perplexity.ai"
    api_key_env = "PERPLEXITY_API_KEY"
