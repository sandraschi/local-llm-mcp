"""OpenAI adapter — pass-through (already OpenAI format)."""

from typing import Any

import httpx

from llm_mcp.gateway.base import BaseLLMAdapter, register_provider


@register_provider("openai")
class OpenAIAdapter(BaseLLMAdapter):
    provider = "openai"
    base_url = "https://api.openai.com/v1"
    api_key_env = "OPENAI_API_KEY"

    def map_params(self, body: dict[str, Any]) -> dict[str, Any]:
        return body

    def transform_response(self, resp_data: dict[str, Any], model: str) -> dict[str, Any]:
        return resp_data

    async def complete(self, body: dict[str, Any], headers: dict[str, Any]) -> dict[str, Any]:
        api_key = self.get_api_key(headers)
        if not api_key:
            raise ValueError("OpenAI API key not set. Set OPENAI_API_KEY env var or pass Authorization header.")
        req_headers = self.build_headers(api_key)
        url = self.build_url(self.get_base_url(headers))

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=body, headers=req_headers)
            resp.raise_for_status()
            return resp.json()
