"""Anthropic adapter — translates OpenAI ChatCompletion -> Anthropic Messages API."""

from typing import Any

import httpx

from llm_mcp.gateway.base import BaseLLMAdapter, register_provider


@register_provider("anthropic")
class AnthropicAdapter(BaseLLMAdapter):
    provider = "anthropic"
    base_url = "https://api.anthropic.com/v1"
    api_key_env = "ANTHROPIC_API_KEY"

    def get_api_key(self, headers: dict[str, str]) -> str:
        import os
        key = os.getenv(self.api_key_env, "")
        if not key:
            key = headers.get("x-api-key", "")
        return key

    def build_headers(self, api_key: str) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }

    def build_url(self, base_url: str) -> str:
        return f"{base_url.rstrip('/')}/messages"

    def map_params(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages", [])
        system = ""
        mapped_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
            else:
                mapped_messages.append({
                    "role": msg["role"],
                    "content": msg.get("content", ""),
                })

        mapped = {
            "model": body.get("model", "claude-sonnet-4-20250514"),
            "messages": mapped_messages,
            "max_tokens": body.get("max_tokens", 4096),
        }
        if system:
            mapped["system"] = system
        if "temperature" in body:
            mapped["temperature"] = body["temperature"]
        if "top_p" in body:
            mapped["top_p"] = body["top_p"]
        if "stop" in body:
            mapped["stop_sequences"] = body["stop"] if isinstance(body["stop"], list) else [body["stop"]]
        return mapped

    def transform_response(self, resp_data: dict[str, Any], model: str) -> dict[str, Any]:
        content = ""
        for block in resp_data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")

        usage = resp_data.get("usage", {})
        return self._openai_chunk(
            model=model,
            choice={
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": resp_data.get("stop_reason", "stop"),
                "logprobs": None,
            },
            usage={
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            },
        )

    async def complete(self, body: dict[str, Any], headers: dict[str, Any]) -> dict[str, Any]:
        api_key = self.get_api_key(headers)
        if not api_key:
            raise ValueError("Anthropic API key not set. Set ANTHROPIC_API_KEY env var or pass x-api-key header.")
        mapped = self.map_params(body)
        req_headers = self.build_headers(api_key)
        url = self.build_url(self.get_base_url(headers))

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=mapped, headers=req_headers)
            resp.raise_for_status()
            data = resp.json()

        return self.transform_response(data, body.get("model", ""))
