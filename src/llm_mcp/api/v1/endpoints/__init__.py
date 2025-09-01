"""API endpoints for the LLM MCP server."""

from .mcp_servers import router as mcp_servers_router
from .models import router as models_router

# List of all routers to be included in the API
__all__ = [
    'mcp_servers_router',
    'models_router'
]

# Re-export the router for easier imports
router = models_router
