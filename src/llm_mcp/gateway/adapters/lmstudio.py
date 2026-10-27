"""LM Studio adapter - OpenAI-compatible local server (0.4.0+).

Auth: Server permission keys (Settings > Server). Reads LM_API_TOKEN /
LMSTUDIO_API_KEY env, Authorization: Bearer header, or x-api-key header
(Anthropic-compat clients send the latter). No key = unauthenticated, which
matches a default local LM Studio install.
"""

import os
from typing import Any

from llm_mcp.gateway.adapters.openai import OpenAIAdapter
from llm_mcp.gateway.base import register_provider


@register_provider("lmstudio")
class LMStudioAdapter(OpenAIAdapter):
    provider = "lmstudio"
    base_url = "http://127.0.0.1:1234/v1"
    api_key_env = "LM_API_TOKEN"

    def get_api_key(self, headers: dict[str, str]) -> str:
        lowered = {str(k).lower(): v for k, v in (headers or {}).items()}
        for env_var in ("LM_API_TOKEN", "LMSTUDIO_API_KEY", self.api_key_env):
            if env_var and os.getenv(env_var, ""):
                return os.getenv(env_var, "")
        auth = lowered.get("authorization", "")
        if auth.startswith("Bearer "):
            return auth[len("Bearer ") :]
        if lowered.get("x-api-key"):
            return lowered["x-api-key"]
        return ""

    def build_headers(self, api_key: str) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            headers["x-api-key"] = api_key
        return headers

    def map_params(self, body: dict[str, Any]) -> dict[str, Any]:
        return body
