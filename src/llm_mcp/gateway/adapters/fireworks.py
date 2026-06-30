"""Fireworks AI adapter — OpenAI-compatible."""

from llm_mcp.gateway.adapters.openai import OpenAIAdapter
from llm_mcp.gateway.base import register_provider


@register_provider("fireworks")
class FireworksAdapter(OpenAIAdapter):
    provider = "fireworks"
    base_url = "https://api.fireworks.ai/inference/v1"
    api_key_env = "FIREWORKS_API_KEY"
