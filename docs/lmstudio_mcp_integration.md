# LM Studio MCP Integration

This document provides comprehensive documentation for integrating the LLM MCP server with LM Studio using the Model Context Protocol (MCP).

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [MCP Server Setup](#mcp-server-setup)
- [LM Studio Configuration](#lm-studio-configuration)
- [Using MCP Tools](#using-mcp-tools)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)
- [Security Considerations](#security-considerations)
- [Development](#development)

## Overview

The Model Context Protocol (MCP) allows LM Studio to interact with external tools and services. This integration enables:

- Access to LLM MCP tools directly from LM Studio
- Seamless interaction with local LLM models
- Extension of LM Studio's capabilities
- Custom tool integration

## Prerequisites

- [LM Studio](https://lmstudio.ai/) v0.3.17 or later
- Python 3.8+ installed
- LLM MCP server installed and running
- Port 8000 available (or configured port)

## MCP Server Setup

### 1. Install Dependencies

Ensure you have the required Python packages:

```bash
pip install fastapi uvicorn httpx
```

### 2. Start the MCP Server

From the project root:

```bash
python -m llm_mcp
```

By default, the server runs on `http://localhost:8000`.

### 3. Verify the Server

Check if the server is running:

```bash
curl http://localhost:8000/mcp
```

You should see a JSON response with MCP server information.

## LM Studio Configuration

### 1. Open LM Studio

Launch LM Studio on your system.

### 2. Access MCP Settings

1. Click on the gear icon (⚙️) in the bottom left
2. Navigate to "Program"
3. Click "Install" > "Edit mcp.json"

### 3. Configure MCP

Replace the contents with:

```json
{
  "name": "LLM MCP Server",
  "version": "1.0.0",
  "description": "LLM MCP Server for local LLM management",
  "servers": [
    {
      "name": "Local LLM MCP",
      "url": "http://localhost:8000/mcp"
    }
  ]
}
```

Save the file and restart LM Studio.

## Using MCP Tools

### Available Tools

#### 1. List Models
- **Name**: `list_models`
- **Description**: List all available LLM models
- **Parameters**: None
- **Example**: `list_models()`

#### 2. Generate Text
- **Name**: `generate_text`
- **Description**: Generate text using a specified model
- **Parameters**:
  - `model` (string): Model name
  - `prompt` (string): Input text
  - `max_tokens` (int, optional): Maximum tokens to generate
- **Example**: `generate_text(model="llama3", prompt="Hello, world!")`

### Using Tools in Chat

1. Start a chat with any model
2. Type `/` to see available MCP tools
3. Select a tool and provide required parameters
4. The tool will execute and return results

## Troubleshooting

### Common Issues

#### MCP Tools Not Showing
- **Solution**:
  1. Verify the MCP server is running
  2. Check the URL in `mcp.json`
  3. Restart LM Studio

#### Connection Refused
- **Solution**:
  ```bash
  # Check if the server is running
  curl http://localhost:8000/mcp
  
  # Check for port conflicts
  netstat -tuln | grep 8000
  ```

#### Invalid Tool Response
- **Solution**:
  1. Check server logs for errors
  2. Verify tool parameters
  3. Ensure the tool is properly registered

## Advanced Configuration

### Custom Port

To use a different port (e.g., 8080):

1. Start the server with:
   ```bash
   python -m llm_mcp --port 8080
   ```

2. Update `mcp.json` in LM Studio:
   ```json
   "url": "http://localhost:8080/mcp"
   ```

### Authentication

For production use, enable authentication:

1. Set environment variables:
   ```bash
   export MCP_AUTH_USERNAME=admin
   export MCP_AUTH_PASSWORD=your_secure_password
   ```

2. Update `mcp.json`:
   ```json
   "auth": {
     "type": "basic",
     "username": "admin",
     "password": "your_secure_password"
   }
   ```

## Security Considerations

1. **Local Network**: By default, the MCP server only accepts local connections
2. **Authentication**: Enable authentication for production use
3. **TLS**: For remote access, configure TLS/SSL
4. **Firewall**: Ensure proper firewall rules are in place

## Development

### Adding New Tools

1. Create a new function in the appropriate module
2. Add the `@mcp.tool()` decorator
3. Document the function with a docstring
4. The tool will be automatically discovered

### Example Tool

```python
@mcp.tool()
async def get_system_info() -> dict:
    """Get system information.
    
    Returns:
        dict: System information
    """
    import platform
    return {
        "system": platform.system(),
        "node": platform.node(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor()
    }
```

### Debugging

To enable debug logging:

```bash
LOG_LEVEL=debug python -m llm_mcp
```

## Support

For additional help:
- [LM Studio Documentation](https://lmstudio.ai/docs)
- [MCP Specification](https://github.com/modelcontextprotocol/spec)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
