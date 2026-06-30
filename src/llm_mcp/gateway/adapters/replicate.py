"""Replicate adapter — OpenAI-compatible."""

from llm_mcp.gateway.adapters.openai import OpenAIAdapter
from llm_mcp.gateway.base import register_provider


@register_provider("replicate")
class ReplicateAdapter(OpenAIAdapter):
    provider = "replicate"
    base_url = "https://api.replicate.com/v1"
    api_key_env = "REPLICATE_API_TOKEN"
