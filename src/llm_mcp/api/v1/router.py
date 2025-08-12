"""API v1 router configuration."""
from fastapi import APIRouter

# Create the main API router
api_router = APIRouter()

# Import and include all endpoint routers
from .endpoints import models, mcp_servers

# Include all endpoint routers
api_router.include_router(models.router, tags=["models"])
api_router.include_router(mcp_servers.router, prefix="/mcp", tags=["mcp-servers"])
