"""MCP Server management service for LLM MCP Server."""
from typing import Dict, List, Optional, Any
import json
import os
from pathlib import Path
import logging
from fastmcp import FastMCP
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

class MCPServerConfig(BaseModel):
    """Configuration for an MCP server."""
    name: str = Field(..., description="Unique name for the MCP server")
    description: str = Field("", description="Description of the MCP server")
    server_type: str = Field(..., description="Type of server (python, node, docker, etc.)")
    config: Dict[str, Any] = Field(default_factory=dict, description="Server-specific configuration")
    enabled: bool = Field(True, description="Whether the server is enabled")

    @validator('name')
    def name_must_be_valid(cls, v):
        """Validate the server name."""
        if not v or not v.strip():
            raise ValueError("Server name cannot be empty")
        if not v.replace('_', '').isalnum():
            raise ValueError("Server name can only contain alphanumeric characters and underscores")
        return v.strip()

class MCPServerManager:
    """Manages MCP server configurations and lifecycle."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the MCP server manager.
        
        Args:
            config_path: Path to the MCP server configuration file
        """
        self.config_path = config_path or os.path.expanduser("~/.config/llm-mcp/servers.json")
        self.servers: Dict[str, MCPServerConfig] = {}
        self._mcp_instances: Dict[str, FastMCP] = {}
        self._load_servers()
    
    def _load_servers(self):
        """Load server configurations from disk."""
        try:
            config_dir = os.path.dirname(self.config_path)
            os.makedirs(config_dir, exist_ok=True)
            
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    self.servers = {
                        name: MCPServerConfig(**config) 
                        for name, config in data.items()
                    }
            logger.info(f"Loaded {len(self.servers)} MCP server configurations")
        except Exception as e:
            logger.error(f"Failed to load MCP server configurations: {e}")
            self.servers = {}
    
    def _save_servers(self):
        """Save server configurations to disk."""
        try:
            config_dir = os.path.dirname(self.config_path)
            os.makedirs(config_dir, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(
                    {name: server.dict() for name, server in self.servers.items()},
                    f,
                    indent=2
                )
        except Exception as e:
            logger.error(f"Failed to save MCP server configurations: {e}")
            raise
    
    async def add_server(self, config: MCPServerConfig) -> MCPServerConfig:
        """Add a new MCP server configuration.
        
        Args:
            config: MCP server configuration
            
        Returns:
            The added server configuration
        """
        if config.name in self.servers:
            raise ValueError(f"Server with name '{config.name}' already exists")
        
        self.servers[config.name] = config
        self._save_servers()
        logger.info(f"Added MCP server: {config.name}")
        return config
    
    async def update_server(self, name: str, config: Dict[str, Any]) -> MCPServerConfig:
        """Update an existing MCP server configuration.
        
        Args:
            name: Name of the server to update
            config: Updated configuration values
            
        Returns:
            The updated server configuration
        """
        if name not in self.servers:
            raise ValueError(f"Server with name '{name}' not found")
        
        # Update only the provided fields
        current = self.servers[name].dict()
        updated_config = {**current, **{k: v for k, v in config.items() if v is not None}}
        self.servers[name] = MCPServerConfig(**updated_config)
        self._save_servers()
        logger.info(f"Updated MCP server: {name}")
        return self.servers[name]
    
    async def delete_server(self, name: str) -> bool:
        """Delete an MCP server configuration.
        
        Args:
            name: Name of the server to delete
            
        Returns:
            True if the server was deleted, False otherwise
        """
        if name not in self.servers:
            return False
        
        # Stop the server if it's running
        if name in self._mcp_instances:
            await self.stop_server(name)
        
        del self.servers[name]
        self._save_servers()
        logger.info(f"Deleted MCP server: {name}")
        return True
    
    async def get_server(self, name: str) -> Optional[MCPServerConfig]:
        """Get an MCP server configuration by name.
        
        Args:
            name: Name of the server to get
            
        Returns:
            The server configuration, or None if not found
        """
        return self.servers.get(name)
    
    async def list_servers(self, enabled_only: bool = False) -> List[MCPServerConfig]:
        """List all MCP server configurations.
        
        Args:
            enabled_only: If True, only return enabled servers
            
        Returns:
            List of server configurations
        """
        if enabled_only:
            return [s for s in self.servers.values() if s.enabled]
        return list(self.servers.values())
    
    async def start_server(self, name: str) -> bool:
        """Start an MCP server.
        
        Args:
            name: Name of the server to start
            
        Returns:
            True if the server was started successfully, False otherwise
        """
        if name in self._mcp_instances:
            logger.warning(f"Server '{name}' is already running")
            return True
            
        server_config = await self.get_server(name)
        if not server_config or not server_config.enabled:
            logger.error(f"Cannot start server '{name}': Not found or disabled")
            return False
        
        try:
            # Create and start the FastMCP instance with stateful functionality
            config = server_config.config.copy()
            
            # Ensure stateful configuration is set if not provided
            if 'state' not in config:
                config.update({
                    'state': {
                        'enabled': True,
                        'persistence': {
                            'enabled': True,
                            'storage_path': f"state/{server_config.name.lower().replace(' ', '_')}"
                        },
                        'max_size_mb': 100
                    },
                    'features': ["stateful_tools", "persistent_state", "tool_caching"]
                })
            
            mcp = FastMCP(name=server_config.name, **config)
            
            # Register tools based on server type
            await self._register_server_tools(mcp, server_config)
            
            # Start the server in the background
            await mcp.start()
            self._mcp_instances[name] = mcp
            logger.info(f"Started MCP server: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start MCP server '{name}': {e}")
            return False
    
    async def stop_server(self, name: str) -> bool:
        """Stop an MCP server.
        
        Args:
            name: Name of the server to stop
            
        Returns:
            True if the server was stopped successfully, False otherwise
        """
        if name not in self._mcp_instances:
            logger.warning(f"Server '{name}' is not running")
            return False
        
        try:
            mcp = self._mcp_instances.pop(name)
            await mcp.stop()
            logger.info(f"Stopped MCP server: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop MCP server '{name}': {e}")
            return False
    
    async def _register_server_tools(self, mcp: FastMCP, config: MCPServerConfig):
        """Register tools for an MCP server based on its type.
        
        Args:
            mcp: The FastMCP instance
            config: The server configuration
        """
        # Default tools for all servers
        @mcp.tool()
        async def get_server_status() -> Dict[str, Any]:
            """Get the current status of this MCP server."""
            return {
                "name": config.name,
                "type": config.server_type,
                "enabled": config.enabled,
                "status": "running"
            }
        
        # Register server-type specific tools
        if config.server_type == "python":
            await self._register_python_tools(mcp, config)
        elif config.server_type == "node":
            await self._register_node_tools(mcp, config)
        elif config.server_type == "docker":
            await self._register_docker_tools(mcp, config)
    
    async def _register_python_tools(self, mcp: FastMCP, config: MCPServerConfig):
        """Register tools for a Python-based MCP server."""
        # Add Python-specific tools here
        pass
    
    async def _register_node_tools(self, mcp: FastMCP, config: MCPServerConfig):
        """Register tools for a Node.js-based MCP server."""
        # Add Node.js-specific tools here
        pass
    
    async def _register_docker_tools(self, mcp: FastMCP, config: MCPServerConfig):
        """Register tools for a Docker-based MCP server."""
        # Add Docker-specific tools here
        pass
    
    async def get_server_status(self, name: str) -> Dict[str, Any]:
        """Get the status of an MCP server.
        
        Args:
            name: Name of the server
            
        Returns:
            Status information about the server
        """
        server_config = await self.get_server(name)
        if not server_config:
            return {"error": f"Server '{name}' not found"}
        
        is_running = name in self._mcp_instances
        
        return {
            "name": name,
            "type": server_config.server_type,
            "enabled": server_config.enabled,
            "status": "running" if is_running else "stopped",
            "config": server_config.config
        }
    
    async def stop_all_servers(self):
        """Stop all running MCP servers."""
        for name in list(self._mcp_instances.keys()):
            await self.stop_server(name)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop_all_servers()

# Global instance
mcp_server_manager = MCPServerManager()
