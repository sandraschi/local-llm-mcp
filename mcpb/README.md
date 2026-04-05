# Local LLM MCP Server

**State-of-the-Art MCP Server for Managing Local and Cloud Large Language Models**

[![FastMCP 2.14.1+](https://img.shields.io/badge/FastMCP-2.14.1+-blue.svg)](https://github.com/jlowin/fastmcp)
[![SOTA Compliant](https://img.shields.io/badge/SOTA-Compliant-green.svg)](https://github.com/sandraschi/mcp-central-docs)

## 🎯 **Overview**

The Local LLM MCP Server provides comprehensive management capabilities for Large Language Models across multiple providers including Ollama, LM Studio, and cloud APIs. This server follows **SOTA (State of the Art)** standards with FastMCP 2.14.1+ compliance and advanced portmanteau tool architecture.

## 🚀 **Key Features**

### **31 Specialized Tools**
- **10 Portmanteau Tools**: Consolidated operations for better UX
- **4 GPU Management Tools**: NVIDIA RTX 4090 monitoring and control
- **6 Core Tools**: Essential model and generation operations
- **1 System Tool**: Health monitoring
- **10 Extensive Help Tools**: 5-level documentation system
- **Multi-Provider Support**: Ollama, LM Studio, vLLM, OpenAI, Anthropic
- **Advanced Fine-tuning**: LoRA, Sparse, and DoRA training
- **Multimodal Capabilities**: Image analysis and generation
- **Real-time Monitoring**: System health and performance metrics

### **Portmanteau Architecture**
Following FastMCP 2.13+ best practices, operations are consolidated into logical groups:

| Tool | Operations | Purpose |
|------|------------|---------|
| `llm_health_tool` | Health checks, monitoring, system info | System management |
| `llm_models_tool` | Model listing, loading, provider management | Model operations |
| `llm_generation_tool` | Text generation, chat, embeddings | Content creation |
| `llm_multimodal_tool` | Image analysis, generation, comparison | Visual AI |
| `llm_finetuning_tool` | LoRA, Sparse, DoRA training | Model customization |

## 📦 **Installation**

### **Claude Desktop (MCPB)**
1. Download the `.mcpb` file from releases
2. Drag and drop into Claude Desktop settings
3. The server will be automatically configured

### **Manual Installation**
```bash
# Clone the repository
git clone https://github.com/sandraschi/local-llm-mcp.git
cd local-llm-mcp

# Install dependencies
pip install -e .

# Install optional components
pip install 'transformers>=4.44.0' 'torch>=2.4.0' 'peft>=0.12.0' 'bitsandbytes>=0.49.0'
```

## 🛠️ **Usage Examples**

### **Basic Text Generation**
```python
result = await generate_text(
    model="llama3",
    prompt="Explain machine learning in simple terms"
)
print(result["data"])
```

### **Portmanteau Operations**
```python
# Health check
health = await llm_health_tool("health_check")

# List available models
models = await llm_models_tool("list_models")

# Generate text with advanced parameters
text = await llm_generation_tool("generate_text",
    model="llama3",
    prompt="Write a Python function",
    temperature=0.1,
    max_tokens=300
)
```

### **Model Management**
```python
# Ollama integration
await ollama_pull_model(model="llama3:8b")
await ollama_load_model(model="llama3:8b")

# Use in generation
result = await generate_text(model="llama3:8b", prompt="Hello!")
```

## 🔧 **Configuration**

### **Environment Variables**
```bash
# API Keys (optional)
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# Model cache directory
export LLM_MCP_CACHE_DIR="/path/to/cache"

# Operation timeout
export LLM_MCP_TIMEOUT="300"
```

### **Provider Setup**

#### **Ollama**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull models
ollama pull llama3:8b
ollama pull mistral:7b
```

#### **LM Studio**
1. Download from https://lmstudio.ai/
2. Load models through the LM Studio interface
3. Server will auto-detect running LM Studio instances

#### **Hugging Face (for Gated Models)**
For gated models like FLUX, Black Forest Labs models, etc.:
```bash
# Add to your .env file
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# OR
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

The token will be automatically used for:
- Downloading gated/private models
- Accessing restricted datasets
- Repository management operations

#### **Google Cloud (for Gemini Models)**
For Gemini 3 Flash, Nano Banana Pro, and other Google Cloud AI models:
```bash
# Add to your .env file
GOOGLE_CLOUD_TOKEN=your-google-ai-api-key
# OR
GEMINI_API_KEY=your-google-ai-api-key
GOOGLE_AI_API_KEY=your-google-ai-api-key

# Optional: For Vertex AI operations
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_REGION=us-central1
GOOGLE_CLOUD_BUCKET=your-cloud-storage-bucket
```

The configuration supports:
- **Gemini API**: Direct API access for Gemini models
- **Vertex AI**: Full Google Cloud AI platform integration
- **Cloud Storage**: Model/dataset storage and management
- **Model Deployment**: Deploy custom models to endpoints

## 📊 **System Requirements**

### **Minimum**
- Python 3.10+
- 8GB RAM
- 10GB free disk space

### **Recommended for Training**
- Python 3.10+
- 32GB RAM
- 100GB SSD storage
- NVIDIA GPU with 8GB+ VRAM (for local models)

### **Dependencies**
- FastMCP 2.14.1+
- PyTorch 2.4.0+
- Transformers 4.44.0+
- Optional: vLLM, PEFT, bitsandbytes

## 🎨 **Architecture**

### **SOTA Compliance Features**
- ✅ **FastMCP 2.14.1+** - Latest protocol version
- ✅ **Portmanteau Pattern** - Consolidated tool interfaces
- ✅ **Enhanced Response Patterns** - Rich AI dialogue support
- ✅ **MCPB Packaging** - Drag-and-drop Claude Desktop integration
- ✅ **Comprehensive Prompts** - Extensive system and user guidance

### **Tool Organization**
```
Portmanteau Tools (10)
├── llm_health_tool - System monitoring & diagnostics
├── llm_models_tool - Model management & providers
├── llm_generation_tool - Text generation & chat
├── llm_multimodal_tool - Image analysis & generation
├── llm_finetuning_tool - LoRA, Sparse, DoRA training
├── llm_ollama_tool - Ollama operations
├── llm_lmstudio_tool - LM Studio operations
├── llm_vllm_tool - vLLM high-performance inference
├── llm_huggingface_tool - Hugging Face model/dataset management (supports gated models like FLUX)
└── llm_google_cloud_tool - Google Cloud AI operations (Gemini 3 Flash, Vertex AI, Cloud Storage)

Extensive Help Tools (10)
├── list_available_tools - Tool discovery with 5 detail levels
├── get_tool_help - Comprehensive tool documentation
├── search_tools - Advanced tool search with relevance scoring
├── get_tool_signature - Function signatures with metadata
├── get_workflow_guides - Complete workflow documentation
├── get_performance_guide - Performance optimization guide
├── get_troubleshooting_guide - Comprehensive troubleshooting
├── get_hardware_requirements - Hardware recommendations
├── get_quick_reference - Essential commands and settings
└── get_integration_guide - External system integration

GPU Management Tools (4)
├── gpu_status - GPU monitoring & statistics
├── gpu_clear_memory - Memory cleanup & defragmentation
├── gpu_optimize - Advanced memory optimization
└── gpu_health_check - Health monitoring & diagnostics

Core Tools (7)
├── list_models - Discover available models
├── get_model_info - Detailed model specifications
├── register_model - Register new models
├── generate_text - Text generation
├── chat_completion - Conversational AI
├── embed_text - Text vectorization
└── health_check - System health monitoring
```

## 🔍 **Troubleshooting**

### **Common Issues**

#### **"Model not found"**
```python
# Check available models
models = await list_models()
print("Available:", [m["name"] for m in models["data"]])
```

#### **Memory issues**
```python
# Monitor system resources
health = await llm_health_tool("health_check")
print(f"Memory usage: {health['system']['memory']['percent']}%")

# Unload unused models
await llm_models_tool("unload_model", model_name="large-model")
```

#### **Slow performance**
- Use smaller models for simple tasks
- Enable GPU acceleration if available
- Monitor system resources during operations

### **Debug Information**
```python
# Get detailed health information
health = await llm_health_tool("server_health")

# Check tool availability
tools = await llm_health_tool("list_tools", detail=2)

# View system logs
metrics = await llm_health_tool("get_metrics", name="system_load")
```

## 📚 **Documentation**

- **API Reference**: Comprehensive tool documentation
- **Examples**: 15+ usage examples with expected outputs
- **Workflows**: End-to-end process guides
- **Best Practices**: Performance and reliability tips

## 🤝 **Contributing**

This project follows SOTA standards from MCP Central Docs:

1. **FastMCP 2.14.1+** compliance
2. **Portmanteau pattern** for tool consolidation
3. **Enhanced response patterns** for AI interaction
4. **Comprehensive docstrings** with examples
5. **MCPB packaging** for easy distribution

### **Development Setup**
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Check formatting
ruff check .
black --check .
```

## 📄 **License**

MIT License - see LICENSE file for details.

## 🙏 **Acknowledgments**

- **FastMCP** framework for the underlying MCP implementation
- **Ollama** and **LM Studio** for local model serving
- **Hugging Face** for the transformers library
- **MCP Central Docs** for SOTA standards and best practices

---

**Built with ❤️ following SOTA MCP standards**
