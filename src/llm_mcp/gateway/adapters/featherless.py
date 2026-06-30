"""Featherless AI adapter — OpenAI-compatible."""

from llm_mcp.gateway.adapters.openai import OpenAIAdapter
from llm_mcp.gateway.base import register_provider


@register_provider("featherless")
class FeatherlessAdapter(OpenAIAdapter):
    provider = "featherless"
    base_url = "https://api.featherless.ai/v1"
    api_key_env = "FEATHERLESS_API_KEY"
