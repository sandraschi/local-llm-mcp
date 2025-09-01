"""LLM MCP API v1 package.

This package contains all the API v1 endpoints and related functionality for the LLM MCP server.
"""

from fastapi import APIRouter
from .endpoints import mcp_servers_router, models_router

# Create the main API v1 router
api_router = APIRouter(prefix="/v1", tags=["v1"])

# Include all endpoint routers
api_router.include_router(mcp_servers_router)
api_router.include_router(models_router)

__all__ = ["api_router"]
