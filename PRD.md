# Local LLM MCP - Product Requirements Document

## 1. Overview

**Project Name**: Local LLM MCP  
**Version**: 1.0.0  
**Last Updated**: September 2025  
**Repository**: [github.com/sandraschi/local-llm-mcp](https://github.com/sandraschi/local-llm-mcp)

### 1.1 Product Vision

Local LLM MCP is a high-performance Model Control Protocol (MCP) server designed to manage and serve local large language models with enterprise-grade features. It provides a standardized interface for interacting with various LLM providers while maintaining privacy and control over your AI infrastructure.

### 1.2 Target Audience

- AI/ML Engineers
- DevOps Teams
- Research Scientists
- Enterprise AI Teams
- Privacy-conscious Organizations

## 2. Features

### 2.1 Core Features

- **Multi-Model Support**: Unified interface for multiple LLM providers
- **High-Performance Inference**: Optimized with vLLM's continuous batching
- **Dual Interface Architecture**:
  - **Stdio Interface**: Primary interface for MCP clients (Claude Desktop, etc.) using JSON-RPC over stdio
  - **HTTP/WebSocket Interface**: Secondary interface for testing, debugging, and monitoring
- **RESTful API**: Standardized endpoints for model interaction
- **WebSocket Support**: Real-time streaming of model outputs
- **Authentication & Authorization**: Secure access control for both interfaces
- **Monitoring & Metrics**: Built-in Prometheus metrics
- **Health Checks**: System and model health monitoring

### 2.2 Technical Specifications

| Category           | Details                                                                 |
|--------------------|-------------------------------------------------------------------------|
| **Framework**      | FastMCP 2.12 (Dual Interface)                                           |
| **Backend**        | vLLM 0.10.1.1                                                          |
| **API**            | Dual Interface: Stdio (MCP clients) & HTTP (Testing/Inspection)         |
| **Authentication** | JWT & API Keys                                                         |
| **Deployment**     | Docker, Kubernetes, Bare Metal                                         |
| **Monitoring**     | Prometheus, Grafana, Structured Logging                                |

## 3. Architecture

### 3.1 High-Level Architecture

```
┌─────────────────┐    ┌───────────────────────────────────────┐    ┌─────────────────┐
│  MCP Clients   │    │                                       │    │                 │
│ (Claude Desktop│◄──►│  ┌─────────────────────────────────┐  │    │   Model Store   │
│  & other MCP   │    │  │        FastMCP 2.12 Server       │  │    │  (vLLM Models)  │
│  consumers)    │    │  │  ┌─────────────┐  ┌───────────┐  │  │    │                 │
└────────────────┘    │  │  │  Stdio     │  │  HTTP/WS   │  │  │    └─────────────────┘
                      │  │  │  Interface │  │  Interface │  │  │             ▲
                      │  │  └─────────────┘  └───────────┘  │  │             │
                      │  │         ▲               ▲         │  │             │
                      │  └─────────┼───────────────┼─────────┘  │             │
                      │            │               │            │             │
                      │  ┌─────────▼───────────────▼─────────┐  │             │
                      │  │        Request Router             │◄─┘
                      │  └─────────────────┬─────────────────┘  │
                      │                    │                    │
                      └────────────────────┼────────────────────┘
                                         │
                               ┌─────────┴─────────┐
                               │  Auth & Rate      │
                               │  Limiting Layer   │
                               └─────────┬─────────┘
                                         │
                               ┌─────────▼─────────┐
                               │  Model Controller  │
                               └─────────┬─────────┘
                                         │
                               ┌─────────▼─────────┐
                               │  vLLM Integration  │
                               └───────────────────┘
```

### 3.2 Interface Details

#### Stdio Interface (Primary)

- **Purpose**: Main interface for MCP client applications
- **Protocol**: JSON-RPC over stdio
- **Authentication**: Process-level security (parent/child process relationship)
- **Use Cases**:
  - Claude Desktop integration
  - Other MCP-compatible clients
  - High-performance, low-latency communication

#### HTTP/WebSocket Interface (Secondary)

- **Purpose**: Testing, debugging, and inspection
- **Protocol**: REST API & WebSocket
- **Authentication**: JWT tokens & API keys
- **Use Cases**:
  - Manual testing with tools like curl/Postman
  - Web-based dashboards
  - System monitoring
  - Development and debugging

### 3.3 vLLM Containerization

vLLM runs in a Docker container with the following specifications:
- **Base Image**: `vllm/vllm-openai:0.10.1`
- **Ports**: 8000 (HTTP server)
- **GPU Support**: Enabled via NVIDIA Container Toolkit
- **Volume Mounts**:
  - Model cache directory for persistence
  - Configuration files

### 3.4 Component Architecture

The system follows a modular architecture with the following components:

1. **API Gateway**
   - Request routing and load balancing
   - Authentication and authorization
   - Rate limiting and quotas
   - Request/response validation

2. **Model Serving**
   - vLLM integration for high-performance inference
   - Model versioning and management
   - Batch processing support
   - GPU optimization

3. **Authentication Service**
   - JWT token management
   - API key generation and validation
   - Role-based access control

4. **Monitoring & Logging**
   - Prometheus metrics
   - Structured logging (JSON)
   - Health checks
   - Performance metrics

## 4. API Specifications

### 4.1 Authentication

```http
POST /api/v1/auth/token
Content-Type: application/json

{
  "api_key": "your-api-key"
}
```

### 4.2 Text Generation

```http
POST /api/v1/generate
Authorization: Bearer <token>
Content-Type: application/json

{
  "model": "llama-2-7b-chat",
  "prompt": "Explain quantum computing",
  "max_tokens": 150,
  "temperature": 0.7
}
```

## 5. Deployment

### 5.1 Prerequisites

- Python 3.10+
- FastMCP 2.12
- vLLM 0.10.1.1 (Docker container required for Windows)
- Docker and Docker Compose
- NVIDIA Container Toolkit (for GPU acceleration)
- 16GB+ RAM (32GB+ recommended for larger models)
- Docker (for containerized deployment)

### 5.2 Quick Start

```bash
# Clone the repository
git clone https://github.com/sandraschi/local-llm-mcp.git
cd local-llm-mcp

# Install dependencies
pip install -r requirements.txt

# Start the server
python -m llm_mcp.main
```

## 6. Roadmap

### Phase 1: Core Functionality (Q4 2025)
- [x] Basic MCP server implementation
- [x] vLLM integration
- [x] REST API endpoints
- [ ] WebSocket support
- [ ] Basic authentication

### Phase 2: Enterprise Features (Q1 2026)
- [ ] Advanced monitoring
- [ ] Multi-GPU support
- [ ] Model versioning
- [ ] API documentation

### Phase 3: Scaling (Q2 2026)
- [ ] Kubernetes operator
- [ ] Auto-scaling
- [ ] Multi-node support
- [ ] Advanced caching

## 7. Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to contribute to this project.

## 8. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 9. Contact

For questions or support, please open an issue on GitHub or contact the maintainers.
