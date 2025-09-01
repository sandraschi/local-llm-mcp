# LLM MCP Integrations

This directory contains documentation and resources for integrating LLM MCP with various tools and platforms.

## Available Integrations

### 1. [Ollama Web UI](./ollama_webui_integration.md)

A feature-rich web interface for interacting with Ollama models.

### 2. [LM Studio MCP](./lmstudio_mcp_integration.md)

Integration with LM Studio using the Model Context Protocol (MCP).

## Getting Started

### 1. Prerequisites

- Docker and Docker Compose
- Python 3.8+
- Ollama (for Ollama Web UI)
- LM Studio v0.3.17+ (for MCP integration)

### 2. Quick Start

```bash
# Start Ollama Web UI
docker compose -f docker-compose.ollama-webui.yml up -d

# Start MCP Server
python -m llm_mcp
```

### 3. Access

- Ollama Web UI: <http://localhost:3000>
- MCP Server: <http://localhost:8000/mcp>

## Documentation

- [Ollama Web UI Integration](./ollama_webui_integration.md)
- [LM Studio MCP Integration](./lmstudio_mcp_integration.md)

## Troubleshooting

Common issues and solutions are documented in each integration's guide.

## Contributing

To add a new integration:

1. Create a new markdown file in this directory
2. Follow the existing documentation structure
3. Update this README with a link to your integration
4. Submit a pull request
