# Ollama Web UI Integration

This document provides comprehensive documentation for integrating and using the Ollama Web UI with the LLM MCP project.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Features](#features)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)
- [Security Considerations](#security-considerations)
- [Updating](#updating)
- [Uninstallation](#uninstallation)

## Overview

The Ollama Web UI provides a user-friendly interface for interacting with Ollama models. This integration allows you to:

- Browse and manage your Ollama models
- Chat with models through a clean web interface
- View conversation history
- Access advanced model parameters
- Use the interface from any device on your local network

## Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop/) installed and running
- [Ollama](https://ollama.ai/) installed and running locally
- At least one model pulled in Ollama (e.g., `ollama pull llama3`)
- Port 3000 available on your system

## Installation

1. Clone the LLM MCP repository (if not already done):

   ```bash
   git clone https://github.com/yourusername/local-llm-mcp.git
   cd local-llm-mcp
   ```

2. Start the Ollama Web UI using Docker Compose:

   ```bash
   docker compose -f docker-compose.ollama-webui.yml up -d
   ```

3. Verify the container is running:

   ```bash
   docker ps
   ```

   You should see a container named `ollama-webui` with status "Up".

## Configuration

### Basic Configuration

The default configuration in `docker-compose.ollama-webui.yml` includes:

- Web UI accessible at: <http://localhost:3000>
- Connects to Ollama at: <http://host.docker.internal:11434>
- Data persistence in Docker volume: `ollama-webui-data`

### Environment Variables

You can customize the following environment variables in the `docker-compose.ollama-webui.yml` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_API_BASE_URL` | `http://host.docker.internal:11434` | URL of the Ollama API |
| `WEBUI_PORT` | `3000` | Port for the web interface |
| `TZ` | `UTC` | Timezone for the container |

### Example Custom Configuration

```yaml
services:
  ollama-webui:
    environment:
      - OLLAMA_API_BASE_URL=http://host.docker.internal:11434
      - WEBUI_PORT=3001
      - TZ=America/New_York
```

## Usage

### Accessing the Web UI

1. Open your web browser and navigate to: <http://localhost:3000>
2. You should see the Ollama Web UI login/signup page
3. Create an account or log in with existing credentials

### Connecting to Ollama

1. Click on the gear icon (⚙️) in the bottom left corner
2. Navigate to "API Configuration"
3. Set the "Ollama API Base URL" to: `http://host.docker.internal:11434`
4. Click "Save"
5. Your Ollama models should now be visible in the model selector

### Using the Chat Interface

1. Select a model from the dropdown menu
2. Type your message in the input box and press Enter
3. The model will generate a response
4. Use the sidebar to manage conversations

## Features

### Model Management

- View all available Ollama models
- Pull new models directly from the UI
- Delete unused models to free up space

### Chat Features

- Markdown rendering
- Code syntax highlighting
- Conversation history
- Multiple chat sessions
- Model parameters adjustment

### User Interface

- Light/Dark theme
- Responsive design (works on mobile devices)
- Keyboard shortcuts
- Export/import conversations

## Troubleshooting

### Common Issues

#### Web UI Not Accessible

- **Symptom**: Cannot access <http://localhost:3000>
- **Solution**:

  ```bash
  # Check if the container is running
  docker ps
  
  # Check container logs
  docker logs ollama-webui
  
  # Check if port 3000 is in use
  netstat -tuln | grep 3000
  ```

#### Cannot Connect to Ollama

- **Symptom**: Error connecting to Ollama API
- **Solution**:

  1. Verify Ollama is running: `ollama list`
  2. Check the API URL in Web UI settings
  3. Try accessing Ollama API directly: `curl http://localhost:11434/api/tags`

#### Performance Issues

- **Symptom**: Slow response times
- **Solution**:

  - Check system resources: `docker stats`
  - Reduce the number of concurrent requests
  - Close unused applications

## Advanced Configuration

### Using a Different Port

To use a different port (e.g., 8080):

1. Edit `docker-compose.ollama-webui.yml`:
   ```yaml
   ports:
     - "8080:8080"
   ```

2. Update the `WEBUI_PORT` environment variable

3. Restart the container:
   ```bash
   docker compose -f docker-compose.ollama-webui.yml down
   docker compose -f docker-compose.ollama-webui.yml up -d
   ```

### Enabling HTTPS

1. Create SSL certificates or obtain them from a certificate authority

2. Mount the certificates in the container:

   ```yaml
   volumes:
     - ./certs:/app/backend/certs
   ```

3. Set environment variables:

   ```yaml
   environment:
     - SSL_CERT_FILE=/app/backend/certs/cert.pem
     - SSL_KEY_FILE=/app/backend/certs/key.pem
   ```

## Security Considerations

1. **Default Credentials**: Change the default admin password after first login
2. **Network Exposure**: By default, the Web UI is only accessible from localhost
3. **Data Persistence**: All data is stored in a Docker volume
4. **API Access**: The Web UI has full access to your Ollama instance

## Updating

To update to the latest version of Ollama Web UI:

```bash
docker compose -f docker-compose.ollama-webui.yml pull
docker compose -f docker-compose.ollama-webui.yml up -d --force-recreate
```

## Uninstallation

To completely remove the Ollama Web UI:

```bash
# Stop and remove the container
docker compose -f docker-compose.ollama-webui.yml down

# Remove the Docker volume (optional, removes all data)
docker volume rm local-llm-mcp_ollama-webui-data

# Remove the Docker image (optional)
docker rmi ghcr.io/open-webui/open-webui:main
```

## Support

For additional help, please refer to:
- [Ollama Web UI Documentation](https://github.com/open-webui/open-webui)
- [Ollama Documentation](https://github.com/jmorganca/ollama)
- [Docker Documentation](https://docs.docker.com/)
