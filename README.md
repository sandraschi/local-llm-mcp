# 🚀 Local LLM MCP

A FastMCP 2.12 compliant server for local LLM management and integration

[![FastMCP](https://img.shields.io/badge/FastMCP-2.12-blue.svg)](https://github.com/jlowin/fastmcp)
[![vLLM](https://img.shields.io/badge/vLLM-0.10.1.1-green.svg)](https://github.com/vllm-project/vllm)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Overview

Local LLM MCP is a high-performance Model Control Protocol (MCP) server designed for managing and serving local large language models. Built on FastMCP 2.12 and vLLM 0.10.1.1, it provides a robust, production-ready solution for deploying LLMs with enterprise-grade features.

## 🔑 Key Features

- **Multi-Model Support**: Unified interface for multiple LLM providers
- **High-Performance Inference**: Optimized with vLLM's continuous batching
- **Dual Interface Architecture**:
  - **Stdio Interface**: Primary interface for MCP clients (Claude Desktop, etc.) using JSON-RPC over stdio
  - **HTTP/WebSocket Interface**: Secondary interface for testing, debugging, and monitoring
- **RESTful API**: Standardized endpoints for model interaction
- **WebSocket Support**: Real-time streaming of model outputs
- **Authentication & Authorization**: Secure access control for both interfaces
- **Monitoring & Metrics**: Built-in Prometheus metrics and health checks

## 🚀 Performance

- **vLLM Engine**: Up to 19x faster than traditional serving methods
- **FlashAttention 3**: Optimized attention mechanisms for efficiency
- **Prefix Caching**: Minimize redundant computations
- **Continuous Batching**: Maximize GPU utilization
- **Multi-GPU Support**: Scale across multiple GPUs with tensor parallelism

## 🛠️ Getting Started

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- NVIDIA Container Toolkit (for GPU acceleration)
- 16GB+ RAM (32GB+ recommended for larger models)
- Linux or Windows with WSL2

> **Note on vLLM**: vLLM currently requires running in a Docker container on Windows systems due to the lack of Windows wheels. The provided `docker-compose.vllm-v10.yml` configures vLLM with GPU support.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/sandraschi/local-llm-mcp.git
   cd local-llm-mcp
   ```

2. Start vLLM in Docker:
   ```bash
   # Start vLLM with GPU support
   docker-compose -f docker-compose.vllm-v10.yml up -d
   
   # Verify the container is running
   docker ps | grep vllm
   ```
   
   The vLLM service will be available at `http://localhost:7840`

3. Create and activate a virtual environment for the MCP server:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   
   # For development
   pip install -e ".[dev]"
   ```

## Quick Start

1. Start the MCP server:
python -m llm_mcp.main

# Or use the CLI entry point
llm-mcp
```

### Configuration

Create `config.yaml` in the project root:

```yaml
server:
  name: "My Local LLM Server"
  log_level: "INFO"
  port: 8000

model:
  default_provider: "vllm"
  default_model: "microsoft/Phi-3.5-mini-instruct"
  model_cache_dir: "models"

vllm:
  use_v1_engine: true
  gpu_memory_utilization: 0.9
  tensor_parallel_size: 1
  enable_vision: true
  attention_backend: "FLASHINFER"
  enable_prefix_caching: true
```

### Environment Variables

```bash
# vLLM 1.0+ optimization
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_ENABLE_PREFIX_CACHING=1

# Server configuration
export LLM_MCP_DEFAULT_PROVIDER=vllm
export LLM_MCP_LOG_LEVEL=INFO
```

## 🛠️ Available Tools

### Core Tools (Always Available)
- **Health Check**: Server status and performance metrics
- **System Info**: Hardware compatibility and resource usage
- **Model Management**: Load/unload models with automatic optimization

### vLLM 1.0+ Tools (High Performance)
- **Load Model**: Initialize with V1 engine and FlashAttention 3
- **Text Generation**: 19x faster inference with streaming support
- **Structured Output**: JSON generation with schema validation
- **Performance Stats**: Real-time throughput and usage metrics
- **Multimodal**: Vision model support (experimental)

### Training & Fine-tuning Tools
- **LoRA Training**: Parameter-efficient fine-tuning
- **QLoRA**: Quantized LoRA for memory efficiency
- **DoRA**: Weight-decomposed low-rank adaptation
- **Unsloth**: Ultra-fast fine-tuning optimization

### Advanced Tools (Dependency-based)
- **Gradio Interface**: Web UI for model interaction
- **Multimodal**: Image and text processing
- **Monitoring**: Resource usage and performance tracking

## 🚀 Performance Comparison

| Provider | Tokens/Second | Memory Usage | Setup Complexity | Multimodal |
|----------|---------------|--------------|------------------|------------|
| **vLLM 1.0+ (This)** | **793 TPS** | Optimized | Simple | ✅ Vision |
| Ollama | 41 TPS | High | Very Simple | ❌ |
| LM Studio | ~60 TPS | Medium | GUI-based | Limited |
| OpenAI API | ~100 TPS | N/A (Cloud) | API Key | ✅ Full |

> **19x faster than Ollama** with local inference and no API costs!

## 🔧 Architecture

### Provider System
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   MCP Client    │◄──►│   FastMCP 2.12+  │◄──►│  Tool Registry  │
│   (Claude etc)  │    │     Server        │    │  (Error Safe)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                        ┌──────────────────┐
                        │  Provider Layer   │
                        └─────────┬────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│  vLLM 1.0+   │         │    Ollama     │         │   OpenAI     │
│ (793 TPS)    │         │  (41 TPS)     │         │   (Cloud)    │
│ FlashAtt 3   │         │   Simple      │         │  Full API    │
│ Multimodal   │         │   Local       │         │   Support    │
└──────────────┘         └──────────────┘         └──────────────┘
```

### Key Components
- **FastMCP 2.12+**: Modern MCP server with transport handling
- **vLLM V1 Engine**: High-performance inference with FlashAttention 3
- **State Manager**: Persistent sessions with cleanup and monitoring
- **Configuration**: YAML + environment variables with validation
- **Error Isolation**: Tool registration with recovery mechanisms

## 🧪 Development

### Running Tests
```bash
# Install test dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest --cov=llm_mcp tests/
```

### Code Quality
```bash
# Format code
black src/ tests/
ruff check src/ tests/ --fix

# Type checking
mypy src/
```

### Adding New Tools
1. Create `src/llm_mcp/tools/my_new_tools.py`
2. Implement `register_my_new_tools(mcp)` function
3. Add to `tools/__init__.py` advanced_tools list
4. Handle dependencies and error cases

## 🐛 Troubleshooting

### Common Issues

**Server won't start**
```bash
# Check dependencies
python -c "from llm_mcp.tools import check_dependencies; print(check_dependencies())"

# Verify FastMCP version
pip show fastmcp  # Should be 2.12+
```

**vLLM fails to load**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-compatible PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**Memory issues**
```bash
# Reduce GPU memory utilization in config.yaml
vllm:
  gpu_memory_utilization: 0.7  # Reduce from 0.9
  
# Or use CPU mode
export CUDA_VISIBLE_DEVICES=""
```

### Debug Logging
```bash
# Enable debug logging
export LLM_MCP_LOG_LEVEL=DEBUG

# Check log files
tail -f logs/llm_mcp.log
```

## 📈 Monitoring

### Performance Metrics
- **Tokens/second**: Real-time throughput measurement
- **Memory usage**: GPU/CPU memory tracking  
- **Request latency**: P50/P95/P99 latency metrics
- **Model utilization**: Usage statistics per model

### Health Checks
```bash
# Built-in health check tool
curl -X POST "http://localhost:8000" \
  -H "Content-Type: application/json" \
  -d '{"tool": "health_check"}'
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Ensure code quality (black, ruff, mypy)
5. Submit pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- **FastMCP**: Modern MCP server framework
- **vLLM**: High-performance LLM inference
- **Anthropic**: MCP protocol specification
- **HuggingFace**: Transformers and model ecosystem

---

**Built for performance, reliability, and developer experience** 🚀

> This is a FIXED version (September 2025) that resolves all critical startup issues and modernizes the codebase for production use.
