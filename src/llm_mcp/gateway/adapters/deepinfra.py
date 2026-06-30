"""DeepInfra adapter — OpenAI-compatible."""

from llm_mcp.gateway.adapters.openai import OpenAIAdapter
from llm_mcp.gateway.base import register_provider


@register_provider("deepinfra")
class DeepInfraAdapter(OpenAIAdapter):
    provider = "deepinfra"
    base_url = "https://api.deepinfra.com/v1/openai"
    api_key_env = "DEEPINFRA_API_KEY"
