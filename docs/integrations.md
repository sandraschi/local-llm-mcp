# LLM MCP Integrations

This document provides instructions for integrating LLM MCP with various tools and platforms.

## Table of Contents

- [Ollama Web UI](#ollama-web-ui)
- [LM Studio MCP](#lm-studio-mcp)
- [Troubleshooting](#troubleshooting)

## Ollama Web UI

Ollama Web UI provides a user-friendly interface for interacting with Ollama models.

### Prerequisites

- Docker and Docker Compose installed
- Ollama running locally (default: `http://localhost:11434`)

### Setup

1. Start the Ollama Web UI:

   ```bash
   docker compose -f docker-compose.ollama-webui.yml up -d
   ```

2. Access the web interface at [http://localhost:3000](http://localhost:3000)

3. Connect to your local Ollama instance:
   - Go to Settings → API Configuration
   - Set `Ollama API Base URL` to `http://host.docker.internal:11434`
   - Save settings

### Features

- Chat interface for Ollama models
- Model management
- Conversation history
- Mobile-responsive design

## LM Studio MCP

LM Studio supports the Model Context Protocol (MCP) for extending its capabilities.

### Prerequisites

- LM Studio v0.3.17 or later
- LLM MCP server running (default: `http://localhost:8000`)

### Setup

1. Ensure your MCP server is running:
   ```bash
   # From the project root
   python -m llm_mcp
   ```

2. In LM Studio:
   - Go to the 'Program' tab in the right sidebar
   - Click 'Install' > 'Edit mcp.json'
   - Copy the contents of `mcp.json` from the project root
   - Save the file and restart LM Studio

3. Verify the integration:
   - Start a chat in LM Studio
   - The MCP tools should be available in the chat interface

### Available Tools
- `list_models`: List all available models
- `generate_text`: Generate text using a specified model

## Troubleshooting

### Ollama Web UI
- **Can't connect to Ollama**: Ensure Ollama is running and accessible at http://localhost:11434
- **Docker issues**: Make sure Docker is running and you have sufficient permissions

### LM Studio MCP
- **MCP tools not showing up**: Check that the MCP server is running and the configuration is correct
- **Connection refused**: Verify the MCP server URL in `mcp.json` matches your server's address

## Development

### Updating MCP Configuration
Edit the `mcp.json` file in the project root to add or modify MCP tools and server configurations.

### Adding New Integrations
To add support for additional tools or platforms:
1. Create a new section in this document
2. Add any necessary configuration files
3. Update the setup script if needed
4. Document the integration process
