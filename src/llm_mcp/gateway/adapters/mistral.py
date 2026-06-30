"""Mistral adapter — almost OpenAI-compatible, minor param mapping."""

from llm_mcp.gateway.adapters.openai import OpenAIAdapter
from llm_mcp.gateway.base import register_provider


@register_provider("mistral")
class MistralAdapter(OpenAIAdapter):
    provider = "mistral"
    base_url = "https://api.mistral.ai/v1"
    api_key_env = "MISTRAL_API_KEY"

    def map_params(self, body):
        mapped = dict(body)
        # Mistral doesn't support max_completion_tokens, use max_tokens
        if "max_completion_tokens" in mapped and "max_tokens" not in mapped:
            mapped["max_tokens"] = mapped.pop("max_completion_tokens")
        return mapped
