"""Tests for the MCP server manager."""
import os
import json
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from src.llm_mcp.services.mcp_server_manager import (
    MCPServerManager,
    MCPServerConfig,
)

# Test data
TEST_SERVER_CONFIG = {
    "test_server": {
        "name": "test_server",
        "description": "Test MCP server",
        "server_type": "python",
        "config": {
            "host": "localhost",
            "port": 8001,
            "log_level": "info"
        },
        "enabled": True
    }
}

# Fixtures
@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for test configurations."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def config_file(temp_config_dir):
    """Create a temporary config file with test data."""
    config_path = os.path.join(temp_config_dir, "servers.json")
    with open(config_path, 'w') as f:
        json.dump(TEST_SERVER_CONFIG, f)
    return config_path

@pytest.fixture
async def manager(config_file):
    """Create an MCPServerManager instance with a test config file."""
    manager = MCPServerManager(config_path=config_file)
    yield manager
    await manager.stop_all_servers()

# Tests
class TestMCPServerManager:
    """Test cases for the MCPServerManager class."""
    
    async def test_load_servers(self, manager):
        """Test loading server configurations from file."""
        # The manager should load servers from the config file during initialization
        servers = await manager.list_servers()
        assert len(servers) == 1
        assert servers[0].name == "test_server"
        assert servers[0].server_type == "python"
        assert servers[0].enabled is True
    
    async def test_add_server(self, manager, temp_config_dir):
        """Test adding a new server configuration."""
        # Create a new server config
        new_server = MCPServerConfig(
            name="new_server",
            description="New test server",
            server_type="node",
            config={"port": 3000},
            enabled=True
        )
        
        # Add the server
        added = await manager.add_server(new_server)
        
        # Verify the server was added
        assert added.name == "new_server"
        assert added.server_type == "node"
        
        # Verify the server is in the list
        servers = await manager.list_servers()
        server_names = [s.name for s in servers]
        assert "new_server" in server_names
        
        # Verify the config file was updated
        with open(manager.config_path, 'r') as f:
            config_data = json.load(f)
        assert "new_server" in config_data
    
    async def test_get_server(self, manager):
        """Test getting a server configuration by name."""
        # Get an existing server
        server = await manager.get_server("test_server")
        assert server is not None
        assert server.name == "test_server"
        
        # Try to get a non-existent server
        server = await manager.get_server("nonexistent")
        assert server is None
    
    async def test_update_server(self, manager):
        """Test updating a server configuration."""
        # Update the server
        updated = await manager.update_server("test_server", {"description": "Updated description"})
        assert updated.description == "Updated description"
        
        # Verify the update is reflected in the list
        server = await manager.get_server("test_server")
        assert server.description == "Updated description"
    
    @pytest.mark.asyncio
    async def test_start_stop_server(self, manager):
        """Test starting and stopping a server."""
        # Patch the FastMCP class to avoid starting a real server
        with patch('fastmcp.FastMCP') as mock_mcp_class:
            mock_mcp = AsyncMock()
            mock_mcp_class.return_value = mock_mcp
            
            # Start the server
            success = await manager.start_server("test_server")
            assert success is True
            
            # Verify the server was started
            status = await manager.get_server_status("test_server")
            assert status["status"] == "running"
            
            # Stop the server
            success = await manager.stop_server("test_server")
            assert success is True
            
            # Verify the server was stopped
            status = await manager.get_server_status("test_server")
            assert status["status"] == "stopped"
    
    async def test_delete_server(self, manager):
        """Test deleting a server configuration."""
        # Delete the server
        success = await manager.delete_server("test_server")
        assert success is True
        
        # Verify the server was deleted
        server = await manager.get_server("test_server")
        assert server is None
        
        # Verify the config file was updated
        with open(manager.config_path, 'r') as f:
            config_data = json.load(f)
        assert "test_server" not in config_data
    
    async def test_enable_disable_server(self, manager):
        """Test enabling and disabling a server."""
        # Disable the server
        await manager.update_server("test_server", {"enabled": False})
        
        # Verify the server is disabled
        server = await manager.get_server("test_server")
        assert server.enabled is False
        
        # Enable the server
        await manager.update_server("test_server", {"enabled": True})
        
        # Verify the server is enabled
        server = await manager.get_server("test_server")
        assert server.enabled is True

# Test the API endpoints
@pytest.mark.asyncio
async def test_api_endpoints(test_client):
    """Test the MCP server management API endpoints."""
    # Create a test client with the FastAPI app
    from src.llm_mcp.main import app
    from fastapi.testclient import TestClient
    
    client = TestClient(app)
    
    # Test listing servers
    response = client.get("/api/v1/mcp/servers")
    assert response.status_code == 200
    servers = response.json()
    assert isinstance(servers, list)
    
    # Test creating a new server
    new_server = {
        "name": "test_api_server",
        "description": "Test API server",
        "server_type": "python",
        "config": {"port": 8002},
        "enabled": True
    }
    
    response = client.post("/api/v1/mcp/servers", json=new_server)
    assert response.status_code == 201
    created_server = response.json()
    assert created_server["name"] == "test_api_server"
    
    # Test getting the server
    response = client.get(f"/api/v1/mcp/servers/test_api_server")
    assert response.status_code == 200
    server = response.json()
    assert server["name"] == "test_api_server"
    
    # Test updating the server
    update_data = {"description": "Updated description"}
    response = client.put("/api/v1/mcp/servers/test_api_server", json=update_data)
    assert response.status_code == 200
    updated_server = response.json()
    assert updated_server["description"] == "Updated description"
    
    # Test starting the server
    with patch('src.llm_mcp.services.mcp_server_manager.MCPServerManager.start_server', 
              new_callable=AsyncMock) as mock_start:
        mock_start.return_value = True
        response = client.post("/api/v1/mcp/servers/test_api_server/start")
        assert response.status_code == 200
        assert response.json()["success"] is True
    
    # Test stopping the server
    with patch('src.llm_mcp.services.mcp_server_manager.MCPServerManager.stop_server', 
              new_callable=AsyncMock) as mock_stop:
        mock_stop.return_value = True
        response = client.post("/api/v1/mcp/servers/test_api_server/stop")
        assert response.status_code == 200
        assert response.json()["success"] is True
    
    # Test getting server status
    response = client.get("/api/v1/mcp/servers/test_api_server/status")
    assert response.status_code == 200
    status_info = response.json()
    assert "status" in status_info
    
    # Test deleting the server
    with patch('src.llm_mcp.services.mcp_server_manager.MCPServerManager.delete_server', 
              new_callable=AsyncMock) as mock_delete:
        mock_delete.return_value = True
        response = client.delete("/api/v1/mcp/servers/test_api_server")
        assert response.status_code == 204

# Run the tests
if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", "-s"]))
