"""LM Studio provider implementation for LLM MCP Server (0.4.0+).

Covers LM Studio 0.4.0 (2026-01-28): headless ``llmster`` daemon, native REST
``/api/v1/*`` (stateful chat, model load/unload/download), OpenAI-compat
``/v1/*`` (chat, completions, embeddings, responses), and Server permission-key
auth. OpenAI-compat stays the default path so the gateway keeps working;
native REST is exposed via extra methods (``native_chat``, ``download_model``).
"""

import json
import logging
import os
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any

import httpx

from ...models.base import (
    BaseProvider,
    ModelCapability,
    ModelMetadata,
    ModelProvider,
    ModelStatus,
)

logger = logging.getLogger(__name__)


class LMStudioProvider(BaseProvider):
    """Provider for LM Studio models."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the LM Studio provider.

        Args:
            config: Configuration dictionary with 'base_url', 'timeout', 'api_key' keys.
                api_key falls back to LM_API_TOKEN / LMSTUDIO_API_KEY env.
        """
        super().__init__(config)
        self.base_url = str(config.get("base_url", "http://localhost:1234")).rstrip("/")
        self.timeout = int(config.get("timeout", 60))
        self.api_key = str(
            config.get("api_key", "") or os.getenv("LM_API_TOKEN", "") or os.getenv("LMSTUDIO_API_KEY", "")
        )
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=float(self.timeout))

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers: dict[str, str] = dict(extra or {})
        if self.api_key and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _request(self, method: str, endpoint: str, **kwargs) -> Any:
        """Make an HTTP request to the LM Studio API."""
        url = f"{self.base_url}{endpoint}"
        kwargs.setdefault("headers", {})
        kwargs["headers"] = self._headers(kwargs["headers"])
        try:
            response = await self.client.request(method, url, **kwargs)
            response.raise_for_status()
            if not response.content:
                return {}
            try:
                return response.json()
            except ValueError:
                return {"raw": response.text}
        except httpx.HTTPStatusError as e:
            error_msg = f"LM Studio API error: {e.response.text}"
            if e.response.status_code == 404:
                return None
            logger.error(error_msg)
            raise Exception(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to connect to LM Studio API: {e!s}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    async def _resolve_model(self, model: str | None) -> str:
        """Resolve the model id to send (real 0.4.0 ids, not the legacy placeholder)."""
        if model:
            return model
        try:
            models = await self.list_models()
            if models:
                return models[0].id
        except Exception:
            pass
        return "local-model"

    async def list_models(self) -> list[ModelMetadata]:  # ty: ignore[invalid-method-override]
        """List available models from LM Studio (/v1/models data array)."""
        try:
            response = await self._request("GET", "/v1/models")
            if not response or "data" not in response:
                return []
            return [self._parse_model_metadata(m) for m in response["data"]]
        except Exception as e:
            logger.error(f"Failed to list LM Studio models: {e!s}")
            return []

    async def get_model(self, model_id: str) -> ModelMetadata | None:  # ty: ignore[invalid-method-override]
        """Get details about a specific model by id or name."""
        models = await self.list_models()
        for model in models:
            if model.id == model_id or model.name == model_id:
                return model
        return None

    def _parse_model_metadata(self, model_data: dict[str, Any]) -> ModelMetadata:
        model_id = str(model_data.get("id", ""))
        model_name = str(model_data.get("name", model_id) or model_id)
        capabilities = [ModelCapability.TEXT_GENERATION, ModelCapability.CHAT]
        lowered = f"{model_id} {model_name}".lower()
        if any(k in lowered for k in ("vision", "clip", "vl", "multimodal", "gemma", "qwen", "gpt-oss")):
            # Heuristic: most current local chat models ship vision or tool-capable builds;
            # embeddings stays a separate capability probed via /v1/embeddings, not assumed.
            capabilities.append(ModelCapability.VISION)
        created = model_data.get("created", "")
        return ModelMetadata(
            id=model_id,
            name=model_name or model_id,
            provider=ModelProvider.LMSTUDIO,
            version=str(model_data.get("version", "1.0")),
            status=ModelStatus.LOADED,  # /v1/models lists resident models
            capabilities=capabilities,
            parameters={
                "owned_by": model_data.get("owned_by", "lmstudio"),
                "permissions": model_data.get("permission", []),
                "context_length": model_data.get("context_length", 0),
            },
            created_at=str(created) if created else None,
            updated_at=None,
        )

    async def generate(self, prompt: str, model: str | None = None, **kwargs) -> AsyncGenerator[str, None]:  # ty: ignore[invalid-method-override]
        """Stream a completion via OpenAI-compat /v1/chat/completions.

        Args:
            prompt: The input prompt (sent as a single user message).
            model: Real LM Studio model id (e.g. openai/gpt-oss-20b). When None,
                resolves to the first resident model.
        """
        model_id = await self._resolve_model(model)
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            **{k: v for k, v in kwargs.items() if v is not None},
        }
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=300.0,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk = line[6:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                        choices = data.get("choices", [])
                        if choices:
                            content = choices[0].get("delta", {}).get("content", "")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse LM Studio response: {chunk}")
        except Exception as e:
            logger.error(f"Error in LM Studio generation: {e!s}")
            raise

    async def chat(self, model_id: str, messages: list[dict[str, str]], **kwargs) -> str:
        """Non-streaming chat via /v1/chat/completions (messages passed through).

        Forwards messages, tools, tool_choice, response_format untouched so tool
        calling and structured output keep working.
        """
        try:
            data: dict[str, Any] = {"model": model_id, "messages": messages, "stream": False, **kwargs}
            response = await self._request("POST", "/v1/chat/completions", json=data, timeout=300.0)
            if response and "choices" in response and len(response["choices"]) > 0:
                return response["choices"][0].get("message", {}).get("content", "")
            return ""
        except Exception as e:
            error_msg = f"Failed to chat with model {model_id}: {e!s}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    async def native_chat(
        self,
        model: str,
        user_input: str,
        previous_response_id: str | None = None,
        context_length: int | None = None,
        integrations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Stateful chat via native REST /api/v1/chat (0.4.0+).

        Returns the full response (response_id, output[], stats). Pass
        previous_response_id to continue without resending history.
        """
        payload: dict[str, Any] = {"model": model, "input": user_input}
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
        if context_length:
            payload["context_length"] = context_length
        if integrations is not None:
            payload["integrations"] = integrations
        result = await self._request("POST", "/api/v1/chat", json=payload, timeout=300.0)
        return result if isinstance(result, dict) else {}

    async def load_model(self, model_id: str, **kwargs) -> ModelMetadata:  # ty: ignore[invalid-method-override]
        """Load a model via native REST, falling back to an existence check.

        0.4.0 endpoint: POST /api/v1/models/load. Older servers ignore it, so a
        failed call degrades to "model exists => treat as loaded".
        """
        payload: dict[str, Any] = {"model": model_id}
        payload.update({k: v for k, v in kwargs.items() if v is not None})
        try:
            await self._request("POST", "/api/v1/models/load", json=payload, timeout=300.0)
        except Exception as e:
            logger.warning(f"Native load failed for {model_id}, falling back to lookup: {e!s}")
        model = await self.get_model(model_id)
        if not model:
            # Server accepted the load but /v1/models has not caught up yet.
            model = ModelMetadata(
                id=model_id,
                name=model_id,
                provider=ModelProvider.LMSTUDIO,
                status=ModelStatus.LOADING,
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT],
            )
            return model
        model.status = ModelStatus.LOADED
        return model

    async def unload_model(self, model_id: str) -> bool:
        """Unload via POST /api/v1/models/unload; True when the server confirms."""
        try:
            result = await self._request("POST", "/api/v1/models/unload", json={"model": model_id})
            return result is not None
        except Exception as e:
            logger.warning(f"Native unload failed for {model_id}: {e!s}")
            return False

    async def pull_model(self, model_name: str) -> dict[str, Any]:
        """Download a model via POST /api/v1/models/download (0.4.0+)."""
        try:
            result = await self._request("POST", "/api/v1/models/download", json={"model": model_name}, timeout=300.0)
            if result is None:
                raise ValueError("download endpoint returned 404")
            return {"status": "ok", "model": model_name, "detail": result}
        except Exception as e:
            error_msg = f"Failed to download model {model_name}: {e!s}"
            logger.error(error_msg)
            return {"status": "error", "model": model_name, "message": error_msg}

    # Back-compat alias used by older callers.
    async def download_model(self, model_name: str) -> dict[str, Any]:
        """Alias for pull_model (native download)."""
        return await self.pull_model(model_name)

    async def get_model_info(self, model_name: str | None = None) -> dict[str, Any]:
        """Return the first matching entry from /v1/models ({} when offline)."""
        try:
            response = await self._request("GET", "/v1/models")
            if response and "data" in response and len(response["data"]) > 0:
                if model_name:
                    for entry in response["data"]:
                        if entry.get("id") == model_name:
                            return entry
                return response["data"][0]
            return {
                "id": "lmstudio-model",
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "lmstudio",
            }
        except Exception as e:
            logger.error(f"Failed to get LM Studio model info: {e!s}")
            return {"id": "lmstudio-model", "error": str(e)}

    async def generate_text(self, model_id: str, prompt: str, **kwargs) -> str:
        """Non-streaming collect from generate."""
        result = ""
        async for chunk in self.generate(prompt, model_id, **kwargs):
            result += chunk
        return result

    async def generate_embeddings(self, model_id: str, texts: list[str], **kwargs) -> list[list[float]]:
        """Embeddings via OpenAI-compat /v1/embeddings."""
        try:
            embeddings: list[list[float]] = []
            for text in texts:
                data = {"model": model_id, "input": text, **kwargs}
                response = await self._request("POST", "/v1/embeddings", json=data, timeout=60.0)
                if response and "data" in response and len(response["data"]) > 0:
                    embeddings.append(response["data"][0]["embedding"])
                else:
                    embeddings.append([])
            return embeddings
        except Exception as e:
            error_msg = f"Failed to generate embeddings with model {model_id}: {e!s}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    @property
    def name(self) -> str:
        """Return the name of the provider."""
        return "lmstudio"

    @property
    def supports_streaming(self) -> bool:
        """Return whether the provider supports streaming responses."""
        return True

    async def close(self):
        """Clean up resources."""
        if hasattr(self, "client"):
            await self.client.aclose()
