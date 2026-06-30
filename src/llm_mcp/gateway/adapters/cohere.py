"""Cohere adapter — custom API format."""

from typing import Any

import httpx

from llm_mcp.gateway.base import BaseLLMAdapter, register_provider


@register_provider("cohere")
class CohereAdapter(BaseLLMAdapter):
    provider = "cohere"
    base_url = "https://api.cohere.com/v2"
    api_key_env = "COHERE_API_KEY"

    def build_url(self, base_url: str) -> str:
        return f"{base_url.rstrip('/')}/chat"

    def map_params(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages", [])
        chat_history = []
        last_msg = ""
        for msg in messages[:-1]:
            chat_history.append({
                "role": msg.get("role", "user"),
                "message": msg.get("content", ""),
            })
        if messages:
            last_msg = messages[-1].get("content", "")

        mapped = {
            "model": body.get("model", "command-r-plus"),
            "message": last_msg,
            "max_tokens": body.get("max_tokens", 4096),
        }
        if chat_history:
            mapped["chat_history"] = chat_history
        if "temperature" in body:
            mapped["temperature"] = body["temperature"]
        if "top_p" in body:
            mapped["p"] = body["top_p"]
        return mapped

    def transform_response(self, resp_data: dict[str, Any], model: str) -> dict[str, Any]:
        text = resp_data.get("text", "")
        usage = resp_data.get("usage", {})
        meta = resp_data.get("meta", {})
        billed = meta.get("billed_units", {})
        return self._openai_chunk(
            model=model,
            choice={
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": resp_data.get("finish_reason", "stop"),
                "logprobs": None,
            },
            usage={
                "prompt_tokens": billed.get("input_tokens", usage.get("input_tokens", 0)),
                "completion_tokens": billed.get("output_tokens", usage.get("output_tokens", 0)),
                "total_tokens": billed.get("input_tokens", 0) + billed.get("output_tokens", 0),
            },
        )

    async def complete(self, body: dict[str, Any], headers: dict[str, Any]) -> dict[str, Any]:
        api_key = self.get_api_key(headers)
        if not api_key:
            raise ValueError("Cohere API key not set. Set COHERE_API_KEY env var.")
        mapped = self.map_params(body)
        req_headers = self.build_headers(api_key)
        url = self.build_url(self.get_base_url(headers))

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=mapped, headers=req_headers)
            resp.raise_for_status()
            data = resp.json()

        return self.transform_response(data, body.get("model", "command-r-plus"))
