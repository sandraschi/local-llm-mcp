"""API endpoints for MCP server management."""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse
import logging

from ....services.mcp_server_manager import mcp_server_manager, MCPServerConfig
from ..models.mcp_servers import (
    MCPServer, MCPServerCreate, MCPServerUpdate,
    MCPServerStatus, MCPServerOperation, MCPServerLogs, 
    MCPServerDiscovery, ServerStatus
)
from ....core.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/mcp-servers", response_model=List[MCPServerStatus])
async def list_mcp_servers(enabled_only: bool = True) -> List[MCPServerStatus]:
    """List all configured MCP servers with their status.
    
    Args:
        enabled_only: If True, only return enabled servers
        
    Returns:
        List of MCP servers with their status
    """
    try:
        servers = await mcp_server_manager.list_servers(enabled_only=enabled_only)
        result = []
        
        for server in servers:
            status_info = await mcp_server_manager.get_server_status(server.name)
            result.append(MCPServerStatus(
                name=server.name,
                type=server.server_type,
                status=ServerStatus.RUNNING if status_info.get("status") == "running" else ServerStatus.STOPPED,
                enabled=server.enabled,
                metrics=status_info.get("metrics", {})
            ))
            
        return result
        
    except Exception as e:
        logger.error(f"Failed to list MCP servers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list MCP servers: {str(e)}"
        )

@router.post("/mcp-servers", response_model=MCPServer, status_code=status.HTTP_201_CREATED)
async def create_mcp_server(server: MCPServerCreate) -> MCPServer:
    """Create a new MCP server configuration.
    
    Args:
        server: MCP server configuration
        
    Returns:
        The created MCP server configuration
    """
    try:
        # Convert to MCPServerConfig
        server_config = MCPServerConfig(
            name=server.name,
            description=server.description,
            server_type=server.server_type,
            config=server.config,
            enabled=server.enabled
        )
        
        created = await mcp_server_manager.add_server(server_config)
        
        # Start the server if enabled
        if server.enabled:
            await mcp_server_manager.start_server(server.name)
        
        return MCPServer(
            **created.dict(),
            status=ServerStatus.RUNNING if server.enabled else ServerStatus.STOPPED
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to create MCP server: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create MCP server: {str(e)}"
        )

@router.get("/mcp-servers/{server_name}", response_model=MCPServer)
async def get_mcp_server(server_name: str) -> MCPServer:
    """Get details about a specific MCP server.
    
    Args:
        server_name: Name of the MCP server
        
    Returns:
        MCP server details and status
    """
    try:
        server = await mcp_server_manager.get_server(server_name)
        if not server:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP server '{server_name}' not found"
            )
            
        status_info = await mcp_server_manager.get_server_status(server_name)
        
        return MCPServer(
            **server.dict(),
            status=ServerStatus.RUNNING if status_info.get("status") == "running" else ServerStatus.STOPPED
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get MCP server '{server_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get MCP server: {str(e)}"
        )

@router.put("/mcp-servers/{server_name}", response_model=MCPServer)
async def update_mcp_server(
    server_name: str,
    server_update: MCPServerUpdate
) -> MCPServer:
    """Update an MCP server configuration.
    
    Args:
        server_name: Name of the MCP server to update
        server_update: Updated configuration values
        
    Returns:
        The updated MCP server configuration
    """
    try:
        # Get current config
        current = await mcp_server_manager.get_server(server_name)
        if not current:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP server '{server_name}' not found"
            )
        
        # Update fields
        update_data = server_update.dict(exclude_unset=True)
        
        # Handle server restart if enabled status changed
        was_enabled = current.enabled
        will_enable = update_data.get('enabled', was_enabled)
        
        # Update the server configuration
        updated = await mcp_server_manager.update_server(server_name, update_data)
        
        # Handle server start/stop based on enabled status
        if was_enabled and not will_enable:
            await mcp_server_manager.stop_server(server_name)
        elif not was_enabled and will_enable:
            await mcp_server_manager.start_server(server_name)
        
        # Get updated status
        status_info = await mcp_server_manager.get_server_status(server_name)
        
        return MCPServer(
            **updated.dict(),
            status=ServerStatus.RUNNING if status_info.get("status") == "running" else ServerStatus.STOPPED
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update MCP server '{server_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update MCP server: {str(e)}"
        )

@router.delete("/mcp-servers/{server_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_mcp_server(server_name: str):
    """Delete an MCP server configuration.
    
    Args:
        server_name: Name of the MCP server to delete
    """
    try:
        success = await mcp_server_manager.delete_server(server_name)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP server '{server_name}' not found"
            )
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete MCP server '{server_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete MCP server: {str(e)}"
        )

@router.post("/mcp-servers/{server_name}/start", response_model=MCPServerOperation)
async def start_mcp_server(server_name: str) -> MCPServerOperation:
    """Start an MCP server.
    
    Args:
        server_name: Name of the MCP server to start
        
    Returns:
        Operation status and server information
    """
    try:
        # Check if server exists and is enabled
        server = await mcp_server_manager.get_server(server_name)
        if not server:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP server '{server_name}' not found"
            )
            
        if not server.enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot start disabled MCP server '{server_name}'"
            )
        
        # Start the server
        success = await mcp_server_manager.start_server(server_name)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to start MCP server '{server_name}'"
            )
        
        # Get updated status
        status_info = await mcp_server_manager.get_server_status(server_name)
        
        return MCPServerOperation(
            success=True,
            message=f"MCP server '{server_name}' started successfully",
            server=MCPServerStatus(
                name=server_name,
                type=server.server_type,
                status=ServerStatus.RUNNING,
                enabled=server.enabled,
                metrics=status_info.get("metrics", {})
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start MCP server '{server_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start MCP server: {str(e)}"
        )

@router.post("/mcp-servers/{server_name}/stop", response_model=MCPServerOperation)
async def stop_mcp_server(server_name: str) -> MCPServerOperation:
    """Stop an MCP server.
    
    Args:
        server_name: Name of the MCP server to stop
        
    Returns:
        Operation status and server information
    """
    try:
        # Check if server exists
        server = await mcp_server_manager.get_server(server_name)
        if not server:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP server '{server_name}' not found"
            )
        
        # Stop the server
        success = await mcp_server_manager.stop_server(server_name)
        if not success:
            # This might happen if the server was already stopped
            logger.warning(f"MCP server '{server_name}' was already stopped or failed to stop")
        
        return MCPServerOperation(
            success=True,
            message=f"MCP server '{server_name}' stopped successfully",
            server=MCPServerStatus(
                name=server_name,
                type=server.server_type,
                status=ServerStatus.STOPPED,
                enabled=server.enabled
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop MCP server '{server_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop MCP server: {str(e)}"
        )

@router.get("/mcp-servers/{server_name}/status", response_model=MCPServerStatus)
async def get_mcp_server_status(server_name: str) -> MCPServerStatus:
    """Get the current status of an MCP server.
    
    Args:
        server_name: Name of the MCP server
        
    Returns:
        Current status and metrics of the MCP server
    """
    try:
        # Check if server exists
        server = await mcp_server_manager.get_server(server_name)
        if not server:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP server '{server_name}' not found"
            )
        
        # Get status
        status_info = await mcp_server_manager.get_server_status(server_name)
        
        return MCPServerStatus(
            name=server_name,
            type=server.server_type,
            status=ServerStatus.RUNNING if status_info.get("status") == "running" else ServerStatus.STOPPED,
            enabled=server.enabled,
            metrics=status_info.get("metrics", {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get status for MCP server '{server_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get MCP server status: {str(e)}"
        )

@router.get("/mcp-servers/{server_name}/logs", response_model=MCPServerLogs)
async def get_mcp_server_logs(
    server_name: str,
    limit: int = 100,
    next_token: Optional[str] = None
) -> MCPServerLogs:
    """Get logs from an MCP server.
    
    Args:
        server_name: Name of the MCP server
        limit: Maximum number of log entries to return
        next_token: Token for pagination
        
    Returns:
        Log entries from the MCP server
    """
    try:
        # Check if server exists
        server = await mcp_server_manager.get_server(server_name)
        if not server:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP server '{server_name}' not found"
            )
        
        # In a real implementation, this would fetch logs from the server
        # For now, return a placeholder response
        return MCPServerLogs(
            server=server_name,
            logs=[],  # Placeholder - implement actual log retrieval
            next_token=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get logs for MCP server '{server_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get MCP server logs: {str(e)}"
        )

@router.get("/mcp-servers/discover/available", response_model=List[MCPServerDiscovery])
async def discover_available_servers() -> List[MCPServerDiscovery]:
    """Discover available MCP servers on the system.
    
    Returns:
        List of discovered MCP servers
    """
    try:
        # In a real implementation, this would scan the system for available MCP servers
        # For now, return a placeholder response
        return []
        
    except Exception as e:
        logger.error(f"Failed to discover available MCP servers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to discover available MCP servers: {str(e)}"
        )
