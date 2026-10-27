"""Unit tests for the LM Studio provider (0.4.0 APIs). All HTTP is mocked; no server needed."""

from unittest.mock import AsyncMock

import pytest

from llm_mcp.models.base import ModelProvider
from llm_mcp.providers.lmstudio.provider import LMStudioProvider


def _provider(**overrides):
    cfg = {"base_url": "http://localhost:1234", "timeout": 60}
    cfg.update(overrides)
    return LMStudioProvider(cfg)


def test_provider_identity():
    p = _provider()
    assert p.name == "lmstudio"
    assert p.supports_streaming is True
    assert p.base_url == "http://localhost:1234"


def test_auth_header_from_config():
    p = _provider(api_key="secret-123")
    assert p._headers() == {"Authorization": "Bearer secret-123"}


def test_auth_header_empty_without_key(monkeypatch):
    monkeypatch.delenv("LM_API_TOKEN", raising=False)
    monkeypatch.delenv("LMSTUDIO_API_KEY", raising=False)
    p = _provider()
    assert p._headers() == {}


@pytest.mark.asyncio
async def test_list_models_parses_all_entries():
    p = _provider()
    p._request = AsyncMock(
        return_value={
            "data": [
                {"id": "openai/gpt-oss-20b", "owned_by": "org"},
                {"id": "qwen3-8b", "owned_by": "lmstudio"},
            ]
        }
    )
    models = await p.list_models()
    assert [m.id for m in models] == ["openai/gpt-oss-20b", "qwen3-8b"]
    assert all(m.provider == ModelProvider.LMSTUDIO for m in models)


@pytest.mark.asyncio
async def test_chat_passes_messages_through_untouched():
    p = _provider()
    seen = {}

    async def fake_request(method, endpoint, **kwargs):
        seen["endpoint"] = endpoint
        seen["json"] = kwargs["json"]
        return {"choices": [{"message": {"content": "hi"}}]}

    p._request = fake_request  # type: ignore[assignment]
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    out = await p.chat(
        "openai/gpt-oss-20b",
        messages,
        tools=[{"type": "function", "function": {"name": "t"}}],
    )
    assert out == "hi"
    assert seen["endpoint"] == "/v1/chat/completions"
    assert seen["json"]["messages"] == messages
    assert seen["json"]["model"] == "openai/gpt-oss-20b"
    assert "tools" in seen["json"]


@pytest.mark.asyncio
async def test_load_model_uses_native_endpoint_with_fallback():
    p = _provider()
    calls = []

    async def fake_request(method, endpoint, **kwargs):
        calls.append(endpoint)
        if endpoint == "/api/v1/models/load":
            return {"ok": True}
        if endpoint == "/v1/models":
            return {"data": [{"id": "openai/gpt-oss-20b"}]}
        raise AssertionError(endpoint)

    p._request = fake_request  # type: ignore[assignment]
    model = await p.load_model("openai/gpt-oss-20b")
    assert calls[0] == "/api/v1/models/load"
    assert model.id == "openai/gpt-oss-20b"


@pytest.mark.asyncio
async def test_generate_embeddings_posts_per_text():
    p = _provider()
    posted = []

    async def fake_request(method, endpoint, **kwargs):
        posted.append((endpoint, kwargs["json"]))
        return {"data": [{"embedding": [0.1, 0.2]}]}

    p._request = fake_request  # type: ignore[assignment]
    out = await p.generate_embeddings("emb-model", ["a", "b"])
    assert out == [[0.1, 0.2], [0.1, 0.2]]
    assert all(e == "/v1/embeddings" for e, _ in posted)
    assert posted[0][1]["model"] == "emb-model"
