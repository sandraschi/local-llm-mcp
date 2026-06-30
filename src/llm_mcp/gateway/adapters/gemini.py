"""Gemini adapter — translates OpenAI ChatCompletion -> Gemini generateContent."""

from typing import Any

import httpx

from llm_mcp.gateway.base import BaseLLMAdapter, register_provider


@register_provider("gemini")
class GeminiAdapter(BaseLLMAdapter):
    provider = "gemini"
    base_url = "https://generativelanguage.googleapis.com/v1beta"
    api_key_env = "GEMINI_API_KEY"

    def get_api_key(self, headers: dict[str, str]) -> str:
        import os
        return os.getenv(self.api_key_env, "")

    def build_headers(self, api_key: str) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def build_url(self, base_url: str) -> str:
        return f"{base_url.rstrip('/')}/models/gemini-2.0-flash:generateContent"

    def get_base_url(self, headers: dict[str, str]) -> str:
        import os
        key = os.getenv(self.api_key_env, "")
        base = os.getenv("GEMINI_BASE_URL", self.base_url)
        return f"{base}?key={key}" if key else base

    def map_params(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages", [])
        contents = []
        system = ""
        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
            else:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})

        mapped: dict[str, Any] = {"contents": contents}
        if system:
            mapped["systemInstruction"] = {"parts": [{"text": system}]}
        gen_conf: dict[str, Any] = {}
        if "temperature" in body:
            gen_conf["temperature"] = body["temperature"]
        if "max_tokens" in body:
            gen_conf["maxOutputTokens"] = body["max_tokens"]
        if "top_p" in body:
            gen_conf["topP"] = body["top_p"]
        if gen_conf:
            mapped["generationConfig"] = gen_conf
        return mapped

    def transform_response(self, resp_data: dict[str, Any], model: str) -> dict[str, Any]:
        candidates = resp_data.get("candidates", [])
        text = ""
        finish = "stop"
        if candidates:
            c = candidates[0]
            content = c.get("content", {})
            parts = content.get("parts", [])
            for p in parts:
                text += p.get("text", "")
            finish_reason = c.get("finishReason", "")
            finish = {
                "STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter",
                "RECITATION": "content_filter", "OTHER": "stop",
            }.get(finish_reason, "stop")

        usage = resp_data.get("usageMetadata", {})
        return self._openai_chunk(
            model=model,
            choice={
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish,
                "logprobs": None,
            },
            usage={
                "prompt_tokens": usage.get("promptTokenCount", 0),
                "completion_tokens": usage.get("candidatesTokenCount", 0),
                "total_tokens": usage.get("totalTokenCount", 0),
            },
        )

    async def complete(self, body: dict[str, Any], headers: dict[str, Any]) -> dict[str, Any]:
        api_key = self.get_api_key(headers)
        if not api_key:
            raise ValueError("Gemini API key not set. Set GEMINI_API_KEY env var.")
        mapped = self.map_params(body)
        req_headers = self.build_headers(api_key)
        url = self.build_url(self.get_base_url(headers))

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=mapped, headers=req_headers)
            resp.raise_for_status()
            data = resp.json()

        return self.transform_response(data, body.get("model", "gemini-2.0-flash"))
