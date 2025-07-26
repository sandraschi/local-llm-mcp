"""Application startup and shutdown handlers."""
from typing import Any, Dict

from fastapi import FastAPI
from fastmcp import FastMCP

from .config import Settings, get_settings
from ..services.model_manager import ModelManager

async def startup_event(app: FastAPI) -> None:
    """Initialize application services on startup."""
    settings = get_settings()
    
    # Initialize model manager
    model_manager = ModelManager(settings.providers.dict())
    
    # Store in app state
    app.state.model_manager = model_manager
    
    # Initialize model manager
    await model_manager.initialize()
    
    print("Application startup complete")

async def shutdown_event(app: FastAPI) -> None:
    """Clean up resources on shutdown."""
    # Clean up model manager
    if hasattr(app.state, 'model_manager'):
        await app.state.model_manager.close()
    
    print("Application shutdown complete")

def setup_mcp(mcp: FastMCP) -> None:
    """Set up FastMCP-specific configurations."""
    settings = get_settings()
    
    # Register tools and resources
    # Example:
    # @mcp.tool()
    # async def list_models() -> List[Dict[str, Any]]:
    #     """List all available models."""
    #     model_manager = mcp.app.state.model_manager
    #     models = await model_manager.list_models()
    #     return [{"id": m.id, "name": m.name, "provider": m.provider.value} for m in models]
    
    # Add more MCP tools here

def register_handlers(app: FastAPI, mcp: FastMCP) -> None:
    """Register startup and shutdown event handlers."""
    @app.on_event("startup")
    async def startup():
        await startup_event(app)
        setup_mcp(mcp)
    
    @app.on_event("shutdown")
    async def shutdown():
        await shutdown_event(app)
