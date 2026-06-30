"""ASGI FastAPI app for the web dashboard backend.

Serves the API used by the web_sota frontend (health, models, MCP servers).
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)


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
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check for the web backend."""
    return {"status": "ok", "service": "llm-mcp-web-api"}


try:
    from llm_mcp.api.v1.router import api_router

    app.include_router(api_router, prefix="/api/v1")
except Exception as e:
    logger.warning("API router not loaded: %s. Dashboard will have limited functionality.", e)

try:
    from llm_mcp.gateway.router import gateway_router

    app.include_router(gateway_router)
    logger.info("Gateway router mounted: %d providers registered", len(__import__("llm_mcp.gateway.base", fromlist=["list_providers"]).list_providers()))
except Exception as e:
    logger.warning("Gateway router not loaded: %s", e)
