"""Lepton AI adapter — OpenAI-compatible."""

from llm_mcp.gateway.adapters.openai import OpenAIAdapter
from llm_mcp.gateway.base import register_provider


@register_provider("lepton")
class LeptonAdapter(OpenAIAdapter):
    provider = "lepton"
    base_url = "https://api.lepton.ai/v1"
    api_key_env = "LEPTON_API_KEY"
