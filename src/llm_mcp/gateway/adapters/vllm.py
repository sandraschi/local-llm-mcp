"""vLLM adapter — OpenAI-compatible local server."""

from llm_mcp.gateway.adapters.openai import OpenAIAdapter
from llm_mcp.gateway.base import register_provider


@register_provider("vllm")
class VLLMAdapter(OpenAIAdapter):
    provider = "vllm"
    base_url = "http://127.0.0.1:8000/v1"
    api_key_env = ""
