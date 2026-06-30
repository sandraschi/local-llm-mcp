"""LM Studio adapter — OpenAI-compatible local server."""

from llm_mcp.gateway.adapters.openai import OpenAIAdapter
from llm_mcp.gateway.base import register_provider


@register_provider("lmstudio")
class LMStudioAdapter(OpenAIAdapter):
    provider = "lmstudio"
    base_url = "http://127.0.0.1:1234/v1"
    api_key_env = ""
