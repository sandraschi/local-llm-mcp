"""Groq adapter — OpenAI-compatible."""

from llm_mcp.gateway.adapters.openai import OpenAIAdapter
from llm_mcp.gateway.base import register_provider


@register_provider("groq")
class GroqAdapter(OpenAIAdapter):
    provider = "groq"
    base_url = "https://api.groq.com/openai/v1"
    api_key_env = "GROQ_API_KEY"
