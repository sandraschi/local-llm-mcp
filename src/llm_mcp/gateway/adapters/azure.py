"""Azure OpenAI adapter — custom endpoint construction."""

from typing import Any

import httpx

from llm_mcp.gateway.adapters.openai import OpenAIAdapter
from llm_mcp.gateway.base import register_provider


@register_provider("azure")
class AzureAdapter(OpenAIAdapter):
    provider = "azure"
    base_url = ""  # Constructed from env vars
    api_key_env = "AZURE_OPENAI_API_KEY"

    def get_base_url(self, headers: dict[str, str]) -> str:
        import os
        resource = os.getenv("AZURE_OPENAI_RESOURCE", "")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        return f"https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions?api-version={api_version}"

    def build_url(self, base_url: str) -> str:
        return base_url

    def build_headers(self, api_key: str) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "api-key": api_key,
        }
