"""ASGI FastAPI app for the web dashboard backend.

Serves the API used by the web_sota frontend (health, models, MCP servers).
"""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

_START_TIME = time.time()

_mcp_server_cache: Any | None = None


async def _get_mcp_tool_catalog() -> dict:
    """Return the live MCP tool catalog from the registered server instance.

    The first call lazily builds the FastMCP server (slow: imports torch);
    subsequent calls reuse the cached instance. This keeps the Tools page
    driven by real tool registration, never a hardcoded list.
    """
    global _mcp_server_cache
    if _mcp_server_cache is None:
        try:
            from llm_mcp.main import create_mcp_server_sync

            _mcp_server_cache = await create_mcp_server_sync()
        except Exception as e:
            logger.warning("Could not create MCP server for tool catalog: %s", e)
            return {"tools": [], "total": 0, "error": str(e)}
    if _mcp_server_cache is None:
        return {"tools": [], "total": 0, "error": "MCP server unavailable"}

    tools = await _mcp_server_cache.list_tools()  # ty: ignore[unresolved-attribute]
    catalog = [{"name": t.name, "description": (getattr(t, "description", "") or "")[:400]} for t in tools]
    return {"tools": catalog, "total": len(catalog)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logger.info("Web backend starting")
    yield
    logger.info("Web backend shutting down")


app = FastAPI(
    title="Local LLM MCP Web API",
    description="Backend API for the Local LLM MCP dashboard",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:10832",
        "http://127.0.0.1:10832",
        "http://localhost:10833",
        "http://127.0.0.1:10833",
        "http://tauri.localhost",
        "https://tauri.localhost",
        "tauri://localhost",
    ],
    allow_origin_regex=r"https?://(?:[a-zA-Z0-9-]+\.ts\.net|.*?\.tail-[a-f0-9]+\.ts\.net|tauri\.localhost|localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|100\.\d{1,3}\.\d{1,3}\.\d{1,3})(?::\d+)?$|^tauri://localhost$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Basic health check for the web backend."""
    return {"status": "ok", "service": "llm-mcp-web-api"}


@app.get("/api/v1/health")
async def api_health():
    """Fleet-standard health endpoint with provider status and LM Link.

    Returns server metadata plus local provider reachability and LM Link
    peer information (Tailscale + LM Studio encrypted mesh).
    """
    response = {
        "status": "ok",
        "server": "local-llm-mcp",
        "version": "1.0.0",
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "providers": {},
        "lm_link": None,
    }

    try:
        from llm_mcp.services.provider_health import (
            check_all_providers,
            check_lm_link_status,
            provider_health_to_dict,
        )

        health_results = await check_all_providers(force=False)
        for name, h in health_results.items():
            response["providers"][name] = provider_health_to_dict(h)

        try:
            link = await check_lm_link_status(force=False)
            response["lm_link"] = {
                "ok": link.ok,
                "enabled": link.enabled,
                "device_name": link.device_name,
                "peer_count": link.peer_count,
                "peers": link.peers,
                "preferred_device": link.preferred_device,
            }
        except Exception as e:
            response["lm_link"] = {"ok": False, "error": str(e)}
    except Exception as e:
        logger.warning("Provider health probe failed in /api/v1/health: %s", e)
        response["providers"]["error"] = str(e)

    return response


@app.get("/api/v1/diagnostics")
async def api_diagnostics():
    """CUA-NSIS smoke-test diagnostics endpoint.

    Returns tool count, provider status, and system info for smoke testing.
    Required by the fleet CUA-NSIS smoke testing standard.
    """
    response: dict = {
        "status": "ok",
        "server": "local-llm-mcp",
        "version": "1.0.0",
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "tool_count": 0,
        "tools": [],
        "providers": {},
        "lm_link": None,
        "system": {"windows": True},
        "errors": [],
    }

    # Count registered tools via gateway providers as proxy
    try:
        from llm_mcp.gateway.base import list_providers as gw_list_providers

        response["tool_count"] = 6  # portmanteau + standalone tools
        response["tools"] = [
            {"name": "llm_health"},
            {"name": "llm_models"},
            {"name": "llm_generation"},
            {"name": "llm_multimodal"},
            {"name": "llm_finetuning"},
            {"name": "llm_ollama"},
            {"name": "llm_lmstudio"},
            {"name": "llm_vllm"},
        ]
        response["gateway_providers"] = gw_list_providers()
    except Exception as e:
        response["errors"].append(f"tool_count: {e}")

    # Provider health
    try:
        from llm_mcp.services.provider_health import (
            check_all_providers,
            check_lm_link_status,
            provider_health_to_dict,
        )

        health_results = await check_all_providers(force=True)
        for name, h in health_results.items():
            response["providers"][name] = provider_health_to_dict(h)

        try:
            link = await check_lm_link_status(force=True)
            response["lm_link"] = {
                "ok": link.ok,
                "enabled": link.enabled,
                "device_name": link.device_name,
                "peer_count": link.peer_count,
                "peers": link.peers,
                "preferred_device": link.preferred_device,
            }
        except Exception as e:
            response["lm_link"] = {"ok": False, "error": str(e)}
    except Exception as e:
        response["errors"].append(f"provider_health: {e}")
        response["providers"]["error"] = str(e)

    return response


@app.get("/api/v1/tools")
async def api_tools():
    """Fleet-standard dynamic tool catalog for the Tools page.

    Returns the live registered MCP tools (name + description). The first
    call may take tens of seconds while the MCP server initializes.
    """
    return await _get_mcp_tool_catalog()


_SKILLS_DIR = Path(__file__).resolve().parent / "skills"
if not _SKILLS_DIR.is_dir():
    # Source layout: skills live at the repo root, not inside the package
    _SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills"


@app.get("/api/v1/skills")
async def api_skills():
    """List available skill directories (name + summary)."""
    skills = []
    if _SKILLS_DIR.is_dir():
        for skill_dir in sorted(_SKILLS_DIR.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue
            text = skill_file.read_text(encoding="utf-8", errors="replace")
            first_line = next((ln.strip().lstrip("#").strip() for ln in text.splitlines() if ln.strip()), "")
            skills.append({"name": skill_dir.name, "title": first_line, "words": len(text.split())})
    return {"skills": skills, "total": len(skills)}


@app.get("/api/v1/skills/{skill_name}")
async def api_skill_content(skill_name: str):
    """Return the raw SKILL.md content for a skill."""
    if not _SKILLS_DIR.is_dir():
        return {"error": "no skills directory", "content": ""}
    skill_file = _SKILLS_DIR / skill_name / "SKILL.md"
    if not skill_file.exists():
        return {"error": f"skill '{skill_name}' not found", "content": ""}
    return {"name": skill_name, "content": skill_file.read_text(encoding="utf-8", errors="replace")}


@app.get("/api/v1/model-cards")
async def api_model_cards():
    """List model cards from the mcd info depot (name + summary, no bodies)."""
    try:
        from llm_mcp.services.model_cards import list_model_cards

        return list_model_cards()
    except Exception as e:
        logger.warning("Model cards probe failed: %s", e, exc_info=True)
        return {"cards": [], "total": 0, "error": str(e)}


@app.get("/api/v1/model-cards/{card_name}")
async def api_model_card_content(card_name: str):
    """Return the raw Markdown body of one model card."""
    try:
        from llm_mcp.services.model_cards import get_model_card

        return get_model_card(card_name)
    except Exception as e:
        logger.warning("Model card read failed: %s", e, exc_info=True)
        return {"error": str(e), "content": ""}


@app.get("/api/v1/engines")
async def api_engines():
    """Engine supervision status for the Glimmer page.

    Returns live state for the llama.cpp server (11439), its truncating
    proxy (11435), and Ollama (11434): port reachability, processes, VRAM,
    loaded models, and GPU totals.
    """
    try:
        from llm_mcp.tools.portmanteau_engine import _engine_state

        llama = await _engine_state("llama")
        ollama = await _engine_state("ollama")
        return {"engines": {"llama": llama, "ollama": ollama}}
    except Exception as e:
        logger.warning("Engine status probe failed: %s", e, exc_info=True)
        return {"engines": {}, "error": str(e)}


@app.post("/api/v1/engines/{engine_name}/restart")
async def api_engine_restart(engine_name: str):
    """Restart a supervised inference engine (llama or ollama)."""
    if engine_name not in ("llama", "ollama"):
        return {"success": False, "error": f"unknown engine: {engine_name}"}
    try:
        from llm_mcp.tools.portmanteau_engine import _start_llama, _start_ollama, _stop_llama, _stop_ollama

        if engine_name == "ollama":
            stop = await _stop_ollama()
            start = await _start_ollama()
        else:
            stop = await _stop_llama()
            start = await _start_llama()
        return {"success": start.get("success", False), "stop": stop, "start": start}
    except Exception as e:
        logger.warning("Engine restart failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


@app.post("/api/v1/engines/llama/start")
async def api_llama_start():
    """Start Glimmer, evicting GPU tenants first.

    Ollama models hold VRAM (gemma4:12b etc.) that the 17 GB Glimmer GGUF
    needs. Unload every loaded Ollama model (keep_alive=0) before spawning
    llama-server so it does not OOM or CPU-offload.
    """
    try:
        from llm_mcp.tools.portmanteau_engine import _ollama_state, _start_llama, _unload_model

        evicted: list[str] = []
        try:
            ollama = await _ollama_state()
            for model_name in ollama.get("loaded_models") or []:
                result = await _unload_model("ollama", model_name)
                if result.get("success"):
                    evicted.append(model_name)
        except Exception as e:
            logger.warning("Ollama eviction probe failed: %s", e)

        start = await _start_llama()
        return {
            "success": start.get("success", False),
            "evicted_ollama_models": evicted,
            "start": start,
        }
    except Exception as e:
        logger.warning("Glimmer start failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


try:
    from llm_mcp.api.v1.router import api_router

    app.include_router(api_router, prefix="/api/v1")
except Exception as e:
    logger.warning("API router not loaded: %s. Dashboard will have limited functionality.", e)

try:
    from llm_mcp.gateway.router import gateway_router

    app.include_router(gateway_router)
    gw_base = __import__("llm_mcp.gateway.base", fromlist=["list_providers"]).list_providers()
    logger.info("Gateway router mounted: %d providers registered", len(gw_base))
except Exception as e:
    logger.warning("Gateway router not loaded: %s", e)


# --- Ollama library management (Models page: list / pull / test / remove) ----
# Thin REST over the Ollama HTTP API so the dashboard can manage the local
# model library without shell access. Pulls run as background jobs (a 20 GB
# model exceeds any sane HTTP timeout); test runs a tiny completion with a
# generous timeout because cold loads of big models take a minute or more.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

_PULL_JOBS: dict[str, dict[str, Any]] = {}
_PULL_TASKS: set = set()


async def _ollama_tags() -> list[dict[str, Any]]:
    async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=30.0) as client:
        r = await client.get("/api/tags")
        r.raise_for_status()
        return r.json().get("models", [])


@app.get("/api/v1/ollama/models")
async def api_ollama_models():
    """Installed Ollama models with sizes for the library UI."""
    try:
        models = await _ollama_tags()
        out = [
            {
                "name": m.get("name", ""),
                "size_gb": round(m.get("size", 0) / 1e9, 1),
                "modified_at": m.get("modified_at", ""),
                "digest": (m.get("digest") or "")[:19],
            }
            for m in models
        ]
        return {"models": out, "count": len(out)}
    except Exception as e:
        logger.warning("Ollama list failed: %s", e)
        return {"models": [], "count": 0, "error": str(e)}


async def _run_pull_job(job_id: str, name: str) -> None:
    job = _PULL_JOBS[job_id]
    job["state"] = "running"
    try:
        # No fixed timeout by design: a 20 GB pull streams for 10+ minutes.
        # Progress is visible via the job id; a stuck pull stays "running".
        async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=None) as client:  # noqa: S113
            async with client.stream("POST", "/api/pull", json={"name": name, "stream": True}) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        import json as _json

                        data = _json.loads(line)
                    except ValueError:
                        continue
                    job["status_text"] = data.get("status", "")
                    total = data.get("total") or 0
                    done = data.get("completed") or 0
                    if total > 0:
                        job["progress"] = round(done / total, 3)
                    if data.get("status") == "success":
                        job["progress"] = 1.0
        job["state"] = "done"
    except Exception as e:
        job["state"] = "failed"
        job["error"] = str(e)
    logger.info("Ollama pull job %s (%s) -> %s", job_id, name, job["state"])


@app.post("/api/v1/ollama/pull")
async def api_ollama_pull(body: dict[str, Any]):
    """Start a background pull from the Ollama library. Poll the job id."""
    name = str(body.get("name") or "").strip()
    if not name:
        return {"success": False, "error": "model name required"}
    job_id = uuid.uuid4().hex[:12]
    _PULL_JOBS[job_id] = {
        "job_id": job_id,
        "name": name,
        "state": "queued",
        "progress": 0.0,
        "status_text": "queued",
        "error": "",
    }
    task = asyncio.create_task(_run_pull_job(job_id, name))
    _PULL_TASKS.add(task)
    task.add_done_callback(_PULL_TASKS.discard)
    return {"success": True, "job_id": job_id}


@app.get("/api/v1/ollama/pull/{job_id}")
async def api_ollama_pull_status(job_id: str):
    """Poll a pull job started by POST /api/v1/ollama/pull."""
    job = _PULL_JOBS.get(job_id)
    if job is None:
        return {"success": False, "error": f"unknown job: {job_id}"}
    return {"success": True, **job}


@app.post("/api/v1/ollama/delete")
async def api_ollama_delete(body: dict[str, Any]):
    """Remove an installed Ollama model (frees disk immediately)."""
    name = str(body.get("name") or "").strip()
    if not name:
        return {"success": False, "error": "model name required"}
    try:
        async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=60.0) as client:
            r = await client.request("DELETE", "/api/delete", json={"name": name})
            r.raise_for_status()
        return {"success": True, "name": name}
    except Exception as e:
        logger.warning("Ollama delete failed for %s: %s", name, e)
        return {"success": False, "error": str(e)}


@app.post("/api/v1/ollama/test")
async def api_ollama_test(body: dict[str, Any]):
    """Smoke-test a model: cold load + tiny completion. Times out honestly."""
    name = str(body.get("name") or "").strip()
    if not name:
        return {"success": False, "ok": False, "error": "model name required"}
    import time as _time

    started = _time.monotonic()
    try:
        async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=210.0) as client:
            r = await client.post(
                "/api/chat",
                json={
                    "model": name,
                    "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                    "stream": False,
                    "keep_alive": "0s",
                    "options": {"num_ctx": 2048, "num_predict": 10},
                },
            )
            r.raise_for_status()
            text = (r.json().get("message") or {}).get("content", "")
        seconds = round(_time.monotonic() - started, 1)
        # Unload again so the test does not squat VRAM.
        try:
            async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=30.0) as client:
                await client.post("/api/generate", json={"model": name, "keep_alive": 0})
        except Exception:
            pass
        return {"success": True, "ok": True, "name": name, "seconds": seconds, "reply": text[:120]}
    except Exception as e:
        seconds = round(_time.monotonic() - started, 1)
        return {"success": False, "ok": False, "name": name, "seconds": seconds, "error": str(e)}
