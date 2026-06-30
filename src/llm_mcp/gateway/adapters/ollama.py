"""Ollama adapter — translates OpenAI ChatCompletion -> Ollama /api/chat."""

from typing import Any

from llm_mcp.gateway.base import BaseLLMAdapter, register_provider


@register_provider("ollama")
class OllamaAdapter(BaseLLMAdapter):
    provider = "ollama"
    base_url = "http://127.0.0.1:11434"
    api_key_env = ""

    def get_base_url(self, headers: dict[str, str]) -> str:
        import os
        return os.getenv("OLLAMA_BASE_URL", self.base_url)

    def get_api_key(self, headers: dict[str, str]) -> str:
        return ""

    def build_url(self, base_url: str) -> str:
        return f"{base_url.rstrip('/')}/api/chat"

    def build_headers(self, api_key: str) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def map_params(self, body: dict[str, Any]) -> dict[str, Any]:
        mapped = {
            "model": body.get("model", ""),
            "messages": body.get("messages", []),
            "stream": False,
        }
        if "temperature" in body:
            mapped["options"] = mapped.get("options", {})
            mapped["options"]["temperature"] = body["temperature"]
        if "max_tokens" in body:
            mapped["options"] = mapped.get("options", {})
            mapped["options"]["num_predict"] = body["max_tokens"]
        if "top_p" in body:
            mapped["options"] = mapped.get("options", {})
            mapped["options"]["top_p"] = body["top_p"]
        return mapped

    def transform_response(self, resp_data: dict[str, Any], model: str) -> dict[str, Any]:
        message = resp_data.get("message", {})
        content = message.get("content", "")
        finish = resp_data.get("done_reason", "stop")
        usage_data = resp_data.get("metrics", {})
        return self._openai_chunk(
            model=model,
            choice={
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish,
                "logprobs": None,
            },
            usage={
                "prompt_tokens": resp_data.get("prompt_eval_count", 0),
                "completion_tokens": resp_data.get("eval_count", 0),
                "total_tokens": (resp_data.get("prompt_eval_count", 0) + resp_data.get("eval_count", 0)),
            },
        )
