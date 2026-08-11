"""Pydantic models for MCP server management API."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class ServerType(StrEnum):
    """Types of MCP servers."""

    PYTHON = "python"
    NODE = "node"
    DOCKER = "docker"
    EXECUTABLE = "executable"
    UNKNOWN = "unknown"


class ServerStatus(StrEnum):
    """Status of an MCP server."""

    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    STARTING = "starting"
    STOPPING = "stopping"


class MCPServerBase(BaseModel):
    """Base model for MCP server configurations."""

    name: str = Field(..., description="Unique name for the MCP server")
    description: str = Field("", description="Description of the MCP server")
    server_type: ServerType = Field(..., description="Type of MCP server")
    config: dict[str, Any] = Field(default_factory=dict, description="Server-specific configuration")
    enabled: bool = Field(True, description="Whether the server is enabled")


class MCPServerCreate(MCPServerBase):
    """Model for creating a new MCP server."""

    pass


class MCPServerUpdate(BaseModel):
    """Model for updating an existing MCP server."""

    description: str | None = Field(None, description="Updated description")
    config: dict[str, Any] | None = Field(None, description="Updated configuration")
    enabled: bool | None = Field(None, description="Whether the server is enabled")


class MCPServer(MCPServerBase):
    """Complete MCP server model with status information."""

    status: ServerStatus = Field(ServerStatus.STOPPED, description="Current server status")
    last_error: str | None = Field(None, description="Last error message, if any")

    class Config:
        """Pydantic config."""

        json_encoders = {ServerType: lambda v: v.value, ServerStatus: lambda v: v.value}


class MCPServerStatus(BaseModel):
    """Status information for an MCP server."""

    name: str = Field(..., description="Server name")
    status: ServerStatus = Field(..., description="Current status")
    type: ServerType = Field(..., description="Server type")
    enabled: bool = Field(..., description="Whether the server is enabled")
    uptime_seconds: float | None = Field(None, description="Uptime in seconds")
    last_error: str | None = Field(None, description="Last error message, if any")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Server metrics and statistics")


class MCPServerList(BaseModel):
    """List of MCP servers with status information."""

    servers: list[MCPServerStatus] = Field(default_factory=list, description="List of MCP servers with their status")


class MCPServerOperation(BaseModel):
    """Response model for MCP server operations."""

    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Status message")
    server: MCPServerStatus | None = Field(None, description="Updated server status")
    error: str | None = Field(None, description="Error details if the operation failed")


class MCPServerLogs(BaseModel):
    """Log entries from an MCP server."""

    server: str = Field(..., description="Server name")
    logs: list[dict[str, Any]] = Field(
        default_factory=list, description="List of log entries with timestamp and message"
    )
    next_token: str | None = Field(None, description="Token for pagination to get the next set of logs")


class MCPServerDiscovery(BaseModel):
    """Information about discovered MCP servers."""

    name: str = Field(..., description="Server name")
    type: ServerType = Field(..., description="Server type")
    description: str = Field("", description="Server description")
    endpoint: HttpUrl | None = Field(None, description="Server endpoint URL if available")
    config: dict[str, Any] = Field(default_factory=dict, description="Suggested configuration for the server")
