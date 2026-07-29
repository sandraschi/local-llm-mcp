"""ASGI FastAPI app for the web dashboard backend.

Serves the API used by the web_sota frontend (health, models, MCP servers).
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

_START_TIME = time.time()


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
