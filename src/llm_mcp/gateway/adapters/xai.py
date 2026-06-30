"""xAI (Grok) adapter — OpenAI-compatible."""

from llm_mcp.gateway.adapters.openai import OpenAIAdapter
from llm_mcp.gateway.base import register_provider


@register_provider("xai")
class XAIAdapter(OpenAIAdapter):
    provider = "xai"
    base_url = "https://api.x.ai/v1"
    api_key_env = "XAI_API_KEY"
