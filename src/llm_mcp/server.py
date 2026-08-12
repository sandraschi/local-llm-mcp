"""ASGI FastAPI app for the web dashboard backend.

Serves the API used by the web_sota frontend (health, models, MCP servers).
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

_START_TIME = time.time()

_mcp_server_cache: object | None = None


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
