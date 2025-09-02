# vLLM v0.10.1.1 + MCP Docker Setup Guide

## 🚀 Quick Start

### Prerequisites
- Docker Desktop with WSL2 backend
- NVIDIA GPU with 8GB+ VRAM (RTX 4090 optimized)
- NVIDIA Container Toolkit
- Git

### 1. Setup Environment
```bash
# Copy environment template
cp docker/.env.example docker/.env

# Edit docker/.env to match your GPU configuration
# Key settings for RTX 4090:
# CUDA_VISIBLE_DEVICES=0
# VLLM_GPU_MEMORY_UTILIZATION=0.9
# TORCH_CUDA_ARCH_LIST=8.9
```

### 2. Start Services
```bash
# Start both vLLM and MCP services
docker-compose -f docker-compose.vllm-v10.yml up -d

# Or use the management script
./docker/scripts/docker-manage.sh start
```

### 3. Verify Installation
```bash
# Check service status
./docker/scripts/docker-manage.sh status

# Test vLLM API
curl http://localhost:8000/v1/models

# Test MCP server
curl http://localhost:3001/health
```

## 📋 Services

### vLLM v0.10.1.1 Server
- **Port**: 8000
- **API**: OpenAI-compatible
- **Model**: Llama 3.1 8B Instruct (default)
- **GPU**: RTX 4090 optimized with FlashInfer
- **Engine**: V1 with prefix caching

### MCP Server
- **Port**: 3001
- **Purpose**: Multi-provider LLM management
- **Features**: Auto-discovery, health monitoring, metrics

## 🔧 Management Commands

### Using Scripts
```bash
# Bash (Linux/WSL)
./docker/scripts/docker-manage.sh [command]

# PowerShell (Windows)
.\docker\scripts\docker-manage.ps1 [command]
```

### Available Commands
- `start` - Start all services
- `stop` - Stop all services  
- `restart` - Restart services
- `status` - Show status and health checks
- `logs` - View recent logs
- `clean` - Stop and cleanup volumes
- `pull` - Update base images
- `rebuild` - Rebuild from scratch

### Direct Docker Compose
```bash
# Start services
docker-compose -f docker-compose.vllm-v10.yml up -d

# View logs
docker-compose -f docker-compose.vllm-v10.yml logs -f

# Stop services
docker-compose -f docker-compose.vllm-v10.yml down
```

## 📚 Model Management

### Download Models
```bash
# Using download script
./docker/scripts/download-models.sh download meta-llama/Meta-Llama-3.1-8B-Instruct

# Popular models
./docker/scripts/download-models.sh popular

# List cached models
./docker/scripts/download-models.sh list
```

### Model Locations
- **HuggingFace Cache**: `~/.cache/huggingface` (mounted in container)
- **Local Models**: `./models` (optional local storage)

## 📊 Monitoring

### Performance Monitoring
```bash
# Check all metrics
./docker/scripts/monitor.sh all

# Real-time monitoring
./docker/scripts/monitor.sh monitor

# GPU only
./docker/scripts/monitor.sh gpu
```

### Log Analysis
```bash
# Recent errors
./docker/scripts/monitor.sh errors

# Service logs
docker-compose logs vllm-v10
docker-compose logs local-llm-mcp
```

## 🔧 Configuration

### Environment Variables (.env)
Key settings for RTX 4090:
```env
CUDA_VISIBLE_DEVICES=0
VLLM_GPU_MEMORY_UTILIZATION=0.9
TORCH_CUDA_ARCH_LIST=8.9
VLLM_USE_V1=1
VLLM_ATTENTION_BACKEND=FLASHINFER
```

### vLLM Configuration (docker/config/vllm-config.yaml)
- Model serving settings
- Performance tuning
- GPU optimization

### MCP Configuration (docker/config/mcp-config.json)
- Provider endpoints
- Health check settings
- Feature toggles

## 🛠️ Troubleshooting

### Common Issues

1. **GPU not detected**
   - Verify NVIDIA Container Toolkit installation
   - Check `nvidia-smi` works in container

2. **Out of memory errors**
   - Reduce `VLLM_GPU_MEMORY_UTILIZATION` to 0.8
   - Use smaller model or reduce context length

3. **Slow performance**
   - Verify FlashInfer backend is active
   - Check GPU utilization with monitoring script

4. **Build failures**
   - Use `rebuild` command for clean build
   - Check Docker has enough disk space

### Health Checks
Services include automatic health checks:
- vLLM: `/health` endpoint every 30s
- MCP: `/health` endpoint every 30s

## 🔄 Backup & Restore

### Create Backup
```bash
./docker/scripts/backup.sh create my-backup-name
```

### Restore Configuration
```bash
./docker/scripts/backup.sh restore my-backup-name
```

## 🌐 API Endpoints

### vLLM OpenAI-Compatible API
- **Base URL**: http://localhost:8000
- **Models**: http://localhost:8000/v1/models
- **Completions**: http://localhost:8000/v1/completions
- **Chat**: http://localhost:8000/v1/chat/completions

### MCP Server API
- **Health**: http://localhost:3001/health
- **Providers**: http://localhost:3001/providers
- **Models**: http://localhost:3001/models

## 📈 Performance Optimization

### RTX 4090 Specific
- CUDA architecture: 8.9
- FlashInfer attention backend
- Expandable CUDA memory segments
- Optimized batch sizes

### Memory Settings
- GPU memory utilization: 90%
- Max batched tokens: 8192
- Max sequences: 256
- Context length: 4096 (configurable)

## 🔗 Integration

The Docker setup provides:
- Isolated vLLM v0.10.1.1 service
- MCP server for unified LLM access
- Automatic service discovery
- Health monitoring
- Network isolation
- Volume persistence

Connect your applications to either:
- vLLM directly: http://localhost:8000
- MCP server: http://localhost:3001 (recommended)
