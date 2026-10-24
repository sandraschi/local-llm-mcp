"""Tests for the Ollama library REST endpoints (Models page backend)."""

import json
import time

from fastapi.testclient import TestClient

from llm_mcp.server import app

client = TestClient(app)

_TAGS = {
    "models": [
        {"name": "tiny:test", "size": 2_000_000_000, "modified_at": "2026-10-20T00:00:00Z", "digest": "abc123"},
        {"name": "big:test", "size": 21_000_000_000, "modified_at": "2026-09-01T00:00:00Z", "digest": "def456"},
    ]
}


class _FakeStream:
    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError("err", request=None, response=self)  # type: ignore[arg-type]

    def json(self):
        return self._payload


class _FakeClient:
    """Minimal httpx.AsyncClient stand-in. Patch llm_mcp.server.httpx.AsyncClient."""

    pull_lines: list = []
    chat_text: str = "OK"
    chat_fail: bool = False

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, url, **kwargs):
        if url.endswith("/api/tags"):
            return _FakeResponse(_TAGS)
        raise AssertionError(f"unexpected GET {url}")

    async def post(self, url, **kwargs):
        if url.endswith("/api/chat"):
            if self.chat_fail:
                import httpx

                raise httpx.HTTPError("boom")
            return _FakeResponse({"message": {"content": self.chat_text}})
        if url.endswith("/api/generate"):
            return _FakeResponse({})
        raise AssertionError(f"unexpected POST {url}")

    async def request(self, method, url, **kwargs):
        if url.endswith("/api/delete"):
            return _FakeResponse({})
        raise AssertionError(f"unexpected {method} {url}")

    def stream(self, method, url, **kwargs):
        assert url.endswith("/api/pull")
        return _FakeStream(self.pull_lines)


def _patch_client(monkeypatch, **kwargs):
    for k, v in kwargs.items():
        setattr(_FakeClient, k, v)
    monkeypatch.setattr("llm_mcp.server.httpx.AsyncClient", _FakeClient)


def test_ollama_models_list(monkeypatch):
    _patch_client(monkeypatch)
    r = client.get("/api/v1/ollama/models")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert body["models"][0]["name"] == "tiny:test"
    assert body["models"][0]["size_gb"] == 2.0


def test_ollama_models_offline(monkeypatch):
    import httpx

    class _Dead:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            raise httpx.ConnectError("refused")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr("llm_mcp.server.httpx.AsyncClient", _Dead)
    r = client.get("/api/v1/ollama/models")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 0
    assert "error" in body


def test_ollama_pull_job_lifecycle(monkeypatch):
    lines = [
        json.dumps({"status": "pulling", "completed": 1, "total": 4}),
        json.dumps({"status": "success"}),
    ]
    _patch_client(monkeypatch, pull_lines=lines)
    r = client.post("/api/v1/ollama/pull", json={"name": "tiny:test"})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    deadline = time.time() + 15
    state = ""
    while time.time() < deadline:
        s = client.get(f"/api/v1/ollama/pull/{job_id}").json()
        state = s.get("state", "")
        if state in ("done", "failed"):
            break
        time.sleep(0.2)
    assert state == "done", s


def test_ollama_pull_rejects_empty_name():
    r = client.post("/api/v1/ollama/pull", json={"name": "  "})
    assert r.json()["success"] is False


def test_ollama_delete(monkeypatch):
    _patch_client(monkeypatch)
    r = client.post("/api/v1/ollama/delete", json={"name": "tiny:test"})
    assert r.json() == {"success": True, "name": "tiny:test"}
    r = client.post("/api/v1/ollama/delete", json={"name": ""})
    assert r.json()["success"] is False


def test_ollama_test_ok(monkeypatch):
    _patch_client(monkeypatch, chat_text="OK")
    r = client.post("/api/v1/ollama/test", json={"name": "tiny:test"})
    body = r.json()
    assert body["ok"] is True
    assert body["reply"] == "OK"
    assert body["seconds"] >= 0


def test_ollama_test_failure(monkeypatch):
    _patch_client(monkeypatch, chat_fail=True)
    r = client.post("/api/v1/ollama/test", json={"name": "ghost:model"})
    body = r.json()
    assert body["ok"] is False
    assert "error" in body
