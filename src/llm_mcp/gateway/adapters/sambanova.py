"""SambaNova adapter — OpenAI-compatible."""

from llm_mcp.gateway.adapters.openai import OpenAIAdapter
from llm_mcp.gateway.base import register_provider


@register_provider("sambanova")
class SambanovaAdapter(OpenAIAdapter):
    provider = "sambanova"
    base_url = "https://api.sambanova.ai/v1"
    api_key_env = "SAMBANOVA_API_KEY"
