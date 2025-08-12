# Local LLM Setup Guide for High-End Workstations

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Recommended Models](#recommended-models]
3. [Setup Instructions](#setup-instructions)
4. [Performance Optimization](#performance-optimization)
5. [Integration with IDEs](#integration-with-ides)
6. [Troubleshooting](#troubleshooting)

## Hardware Requirements

### Minimum

- GPU: NVIDIA 3060 (12GB VRAM)
- RAM: 32GB
- Storage: 100GB free space

### Recommended (Your Setup)

- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **RAM**: 64GB
- **CPU**: 16-core (e.g., Ryzen 9 7950X)
- **Storage**: 1TB NVMe SSD

## Recommended Models

### 1. DeepSeek Coder 33B (AWQ)

- **Best for**: General coding, code completion
- **VRAM**: Fits in 24GB with 4-bit quantization
- **Performance**: 75.4% on HumanEval
- **Format**: AWQ (recommended) or GGUF

### 2. CodeLlama 70B

- **Best for**: Complex problem solving
- **VRAM**: Requires 2x 4090s or 4-bit quantization
- **Performance**: 53.7% on HumanEval
- **Special**: 100k context window available

### 3. WizardCoder 33B

- **Best for**: Python development
- **VRAM**: Fits with 4-bit quantization
- **Performance**: 73.2% on HumanEval

## Setup Instructions

### 1. Install text-generation-webui

```bash
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
pip install -r requirements.txt
```

### 2. Download Model

```bash
python download-model.py TheBloke/deepseek-coder-33B-instruct-AWQ
```

### 3. Start Server

```bash
python server.py --model TheBloke_deepseek-coder-33B-instruct-AWQ \
                --loader awq \
                --listen \
                --api \
                --gpu-memory 20
```

## Performance Optimization

### vLLM for Maximum Throughput

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model TheBloke/deepseek-coder-33B-instruct-AWQ \
  --quantization awq \
  --tensor-parallel-size 1
```

### Recommended Quantization

| Model Size | Quantization | VRAM Usage | Quality |
|------------|--------------|------------|----------|
| 7B         | None        | 14GB       | Best     |
| 13B        | 8-bit       | 13GB       | Excellent|
| 33B        | 4-bit AWQ   | 20GB       | Very Good|
| 70B        | 4-bit AWQ   | 40GB       | Good     |

## Integration with IDEs

### Cursor IDE

1. Install Cursor from [cursor.sh](https://www.cursor.sh/)
2. Go to Settings > AI > Local
3. Enable "Use Local Model"
4. Set API URL to `http://localhost:8000/v1`
5. No API key needed

### VS Code

1. Install Continue extension
2. Add to settings.json:

```json
{
  "continue.OPENAI_API_KEY": "EMPTY",
  "continue.OPENAI_API_BASE": "http://localhost:8000/v1"
}
```

## Troubleshooting

### Out of Memory Errors

- Reduce context length (--n_ctx)
- Increase swap space
- Use smaller model or better quantization

### Slow Performance

- Enable CUDA graphs (--cuda_graphs)
- Use Flash Attention 2
- Disable CPU offloading

### Installation Issues

- Use Python 3.10 or later
- Update CUDA drivers
- Check PyTorch version compatibility

## Advanced: Claude Desktop Proxy

### Setup

1. Create a new Python file `claude_proxy.py`:

```python
from fastapi import FastAPI, Request
import httpx

app = FastAPI()

@app.post("/v1/complete")
async def proxy(request: Request):
    data = await request.json()
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": data["prompt"]}],
                "max_tokens": data.get("max_tokens_to_sample", 2000)
            }
        )
        result = response.json()
    return {"completion": result["choices"][0]["message"]["content"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

2. Start the proxy:

```bash
python claude_proxy.py
```

3. Configure Claude Desktop to use `http://localhost:8001`

## Model Comparison Table

| Model | Size | Quantized | VRAM | Speed | Quality |
|-------|------|-----------|------|-------|---------|
| DeepSeek Coder 33B | 33B | 4-bit AWQ | 20GB | ★★★★☆ | ★★★★★ |
| CodeLlama 70B | 70B | 4-bit AWQ | 40GB | ★★☆☆☆ | ★★★★★ |
| WizardCoder 33B | 33B | 4-bit AWQ | 20GB | ★★★★☆ | ★★★★☆ |
| Mistral 7B | 7B | None | 14GB | ★★★★★ | ★★★☆☆ |

## Maintenance

### Updating Models

```bash
cd text-generation-webui
python download-model.py --update
```

### Monitoring

Use `nvidia-smi` to monitor GPU usage:

```bash
watch -n 1 nvidia-smi
```

## LLM MCP Chat Terminal Integration

The LLM MCP Chat Terminal provides a powerful interface for interacting with local LLM models. Here's how to set it up with your local models:

### 1. Prerequisites

- Python 3.8 or higher
- One of the following local model servers running:
  - text-generation-webui with API enabled (`--api` flag)
  - vLLM server
  - Ollama
  - LM Studio with OpenAI-compatible API

### 2. Configuration

Edit or create `~/.config/llm-mcp/terminal.yaml`:

```yaml
providers:
  local_llm:
    base_url: "http://localhost:8000"  # Update with your local server URL
    api_key: ""  # Leave empty for local servers
    models:
      - deepseek-coder-33b
      - codellama-70b
      - wizardcoder-33b

defaults:
  provider: local_llm
  model: deepseek-coder-33b
  temperature: 0.7
  max_tokens: 2048
```

### 3. Starting the Chat Terminal

```bash
# Navigate to the project directory
cd /path/to/llm-mcp

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Start the chat terminal
python tools/run_chat.py
```

### 4. Using the Terminal

```
# List available models
/model list

# Set a specific model
/model deepseek-coder-33b

# Set temperature (0.0 to 1.0)
/config temperature 0.7

# Start chatting!
Hello, how can you help me with coding today?
```

### 5. Advanced Features

#### Using Personas

```
# List available personas
/personas

# Set a persona
/persona research_assistant
```

#### Applying Rulebooks

```
# List available rulebooks
/rulebooks

# Apply a rulebook
/rulebook technical_writing
```

## Support

For issues, please check:

- [text-generation-webui GitHub](https://github.com/oobabooga/text-generation-webui)
- [vLLM Documentation](https://vllm.readthedocs.io/)
- [Hugging Face Models](https://huggingface.co/models)
- [LLM MCP Documentation](./CHAT_TERMINAL.md)
