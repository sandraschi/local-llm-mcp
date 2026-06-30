"""Hyperbolic adapter — OpenAI-compatible."""

from llm_mcp.gateway.adapters.openai import OpenAIAdapter
from llm_mcp.gateway.base import register_provider


@register_provider("hyperbolic")
class HyperbolicAdapter(OpenAIAdapter):
    provider = "hyperbolic"
    base_url = "https://api.hyperbolic.xyz/v1"
    api_key_env = "HYPERBOLIC_API_KEY"
