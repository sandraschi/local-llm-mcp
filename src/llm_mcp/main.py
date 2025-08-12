"""Main application module for the LLM MCP Server."""
import os
import asyncio
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastmcp import FastMCP
from pydantic import BaseModel

from .core.config import Settings, get_settings
from .core.startup import register_handlers
from .api.v1.router import api_router
from .services.mcp_server_manager import mcp_server_manager

# Initialize FastAPI application first
app = FastAPI(
    title="LLM MCP Server",
    description="A FastMCP 2.10-compliant server for managing local and cloud LLMs",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Initialize FastMCP with the FastAPI app
mcp = FastMCP.from_fastapi(
    app=app,
    name="llm-mcp",
    log_level="INFO"
)

# Register startup and shutdown handlers
register_handlers(app, mcp)

# Set up MCP tools
from .core.startup import setup_mcp
setup_mcp(mcp)

# Initialize MCP server manager
@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup."""
    try:
        # Start any enabled MCP servers
        servers = await mcp_server_manager.list_servers(enabled_only=True)
        for server in servers:
            if server.enabled:
                await mcp_server_manager.start_server(server.name)
        
        logger.info(f"Started {len(servers)} MCP server(s) on startup")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    try:
        # Stop all running MCP servers
        await mcp_server_manager.stop_all_servers()
        logger.info("Stopped all MCP servers")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        raise

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Key security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(
    api_key_header: str = Depends(api_key_header),
    settings: Settings = Depends(get_settings),
) -> str:
    """Validate API key from header."""
    if not settings.server.api_keys:
        return ""
    
    if api_key_header in settings.server.api_keys:
        return api_key_header
    
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate API key",
    )

# Include API router
app.include_router(
    api_router,
    prefix="/api/v1",
    dependencies=[Depends(get_api_key)],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

# Note: The FastMCP instance is already integrated with the FastAPI app
# through the FastMCP.from_fastapi() call, so we don't need to mount it separately.

if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "llm_mcp.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=True,
        log_level=settings.server.log_level.lower(),
    )
