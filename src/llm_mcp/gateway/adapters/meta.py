"""Meta Model API adapter - OpenAI-compatible (https://api.meta.ai/v1)."""

from llm_mcp.gateway.adapters.openai import OpenAIAdapter
from llm_mcp.gateway.base import register_provider


@register_provider("meta")
class MetaAdapter(OpenAIAdapter):
    provider = "meta"
    base_url = "https://api.meta.ai/v1"
    api_key_env = "MODEL_API_KEY"
