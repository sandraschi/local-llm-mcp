"""Application startup and shutdown handlers."""

import logging

from fastapi import FastAPI
from fastmcp import FastMCP

logger = logging.getLogger(__name__)


async def startup_event(app: FastAPI) -> None:
    """Initialize application services on startup."""
    logger.info("Application startup complete")


async def shutdown_event(app: FastAPI) -> None:
    """Clean up resources on shutdown."""
    logger.info("Application shutdown complete")


def setup_mcp(mcp: FastMCP) -> None:
    """Set up FastMCP-specific configurations and register tools."""
    logger.info("FastMCP tool registration delegated to tools package")


def register_handlers(app: FastAPI, mcp: FastMCP) -> None:
    """Register startup and shutdown event handlers."""

    @app.on_event("startup")
    async def startup():
        await startup_event(app)
        setup_mcp(mcp)

    @app.on_event("shutdown")
    async def shutdown():
        await shutdown_event(app)
