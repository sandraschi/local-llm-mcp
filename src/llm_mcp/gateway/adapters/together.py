"""Together AI adapter — OpenAI-compatible."""

from llm_mcp.gateway.adapters.openai import OpenAIAdapter
from llm_mcp.gateway.base import register_provider


@register_provider("together")
class TogetherAdapter(OpenAIAdapter):
    provider = "together"
    base_url = "https://api.together.xyz/v1"
    api_key_env = "TOGETHER_API_KEY"
