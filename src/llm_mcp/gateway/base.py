"""Base adapter for LLM provider gateways.

Each adapter:
1. Maps OpenAI ChatCompletion params -> provider native params
2. Sends request to provider API
3. Maps provider response -> OpenAI ChatCompletion format
"""

import time
import uuid
from abc import ABC, abstractmethod
from typing import Any

import httpx


_openai_compat_providers: dict[str, type["BaseLLMAdapter"]] = {}


def register_provider(name: str):
    """Decorator to register a provider adapter."""
    def wrapper(cls):
        _openai_compat_providers[name] = cls
        return cls
    return wrapper


def get_adapter(provider: str) -> "BaseLLMAdapter | None":
    """Get an adapter instance by provider name."""
    cls = _openai_compat_providers.get(provider)
    if cls:
        return cls()
    return None


def list_providers() -> list[str]:
    """List all registered provider names."""
    return list(_openai_compat_providers.keys())


class BaseLLMAdapter(ABC):
    """Base class for LLM provider adapters.

    Subclasses define:
    - provider name
    - default base URL
    - param mapping from OpenAI format
    - request sending
    - response transformation back to OpenAI format
    """

    provider: str = ""
    base_url: str = ""
    api_key_env: str = ""

    def get_base_url(self, headers: dict[str, str]) -> str:
        return self.base_url

    def get_api_key(self, headers: dict[str, str]) -> str:
        env_key = self.api_key_env
        import os
        key = os.getenv(env_key, "")
        if not key:
            key = headers.get("authorization", "").replace("Bearer ", "")
        return key

    @abstractmethod
    def map_params(self, body: dict[str, Any]) -> dict[str, Any]:
        """Map OpenAI ChatCompletion params to provider-native params."""

    @abstractmethod
    def transform_response(self, resp_data: dict[str, Any], model: str) -> dict[str, Any]:
        """Transform provider-native response to OpenAI ChatCompletion format."""

    def build_headers(self, api_key: str) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def build_url(self, base_url: str) -> str:
        return f"{base_url.rstrip('/')}/v1/chat/completions"

    async def complete(self, body: dict[str, Any], headers: dict[str, Any]) -> dict[str, Any]:
        """Execute a chat completion request through this provider."""
        model = body.get("model", "")
        mapped = self.map_params(body)
        api_key = self.get_api_key(headers)
        req_headers = self.build_headers(api_key)
        url = self.build_url(self.get_base_url(headers))

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=mapped, headers=req_headers)
            resp.raise_for_status()
            data = resp.json()

        return self.transform_response(data, model)

    def _openai_chunk(self, model: str, choice: dict[str, Any], usage: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [choice],
            "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
