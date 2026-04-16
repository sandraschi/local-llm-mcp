"""API v1 router configuration."""

from fastapi import APIRouter

from .endpoints import config, mcp_servers, models, telemetry

# Create the main API router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(models.router, tags=["models"])
api_router.include_router(mcp_servers.router, prefix="/mcp", tags=["mcp-servers"])
api_router.include_router(config.router, prefix="/config", tags=["config"])
api_router.include_router(telemetry.router, prefix="/telemetry", tags=["telemetry"])
