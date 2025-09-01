# vLLM Integration Guide

## Overview

This document provides comprehensive information about the vLLM integration in the LLM MCP server. vLLM is a high-performance and memory-efficient inference and serving engine for LLMs.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Advanced Configuration](#advanced-configuration)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [FAQs](#faqs)

## Features

- 🚀 **High Performance**: Optimized for throughput and latency
- ⚡ **Continuous Batching**: Efficient handling of multiple concurrent requests
- 🧠 **Memory Efficient**: Uses PagedAttention for optimal memory usage
- 🔧 **Flexible**: Supports various model architectures and configurations
- 🔄 **Streaming**: Real-time token streaming support

## Installation

1. Ensure you have a CUDA-compatible GPU with drivers installed
2. Install vLLM and its dependencies:
   ```bash
   pip install vllm>=1.0.0
   ```
3. Verify installation:
   ```python
   import vllm
   print(f"vLLM version: {vllm.__version__}")
   ```

## Quick Start

### Loading a Model

```python
# Load a model with default settings
await mcp.vllm_load_model(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    max_model_len=4096,
    gpu_memory_utilization=0.9
)
```

### Generating Text

```python
# Basic generation
response = await mcp.vllm_generate(
    prompt="Explain quantum computing in simple terms",
    temperature=0.7,
    max_tokens=200
)

# Streaming generation
async for chunk in mcp.vllm_generate(
    prompt="Write a short story about AI",
    stream=True,
    temperature=0.8
):
    print(chunk, end="", flush=True)
```

### Unloading a Model

```python
# Unload the current model to free up resources
await mcp.vllm_unload()
```

## API Reference

### vllm_load_model

Loads a vLLM model with the specified configuration.

**Parameters:**
- `model_name` (str): Name or path of the model to load
- `max_model_len` (int): Maximum sequence length for the model
- `gpu_memory_utilization` (float): Fraction of GPU memory to use (0-1)
- `tensor_parallel_size` (int): Number of GPUs to use for tensor parallelism
- `dtype` (str): Data type for model weights (e.g., 'float16', 'bfloat16')

### vllm_generate

Generates text using the loaded vLLM model.

**Parameters:**
- `prompt` (str): Input text prompt
- `temperature` (float): Sampling temperature (0-2)
- `top_p` (float): Nucleus sampling parameter (0-1)
- `max_tokens` (int): Maximum number of tokens to generate
- `stream` (bool): Whether to stream the response
- `stop` (List[str]): List of stop sequences

### vllm_unload

Unloads the currently loaded model and frees GPU memory.

## Advanced Configuration

### Multi-GPU Setup

```python
# Load model across multiple GPUs
await mcp.vllm_load_model(
    model_name="meta-llama/Llama-2-70b-chat-hf",
    tensor_parallel_size=4,  # Use 4 GPUs
    gpu_memory_utilization=0.9
)
```

### Quantization

```python
# Load model with 8-bit quantization
await mcp.vllm_load_model(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    quantization="awq",  # or "squeezellm"
    gpu_memory_utilization=0.9
)
```

## Performance Tuning

### Optimizing Throughput
- Increase `gpu_memory_utilization` (up to 0.95) for better throughput
- Use `tensor_parallel_size` to distribute model across multiple GPUs
- Adjust `max_model_len` based on your typical sequence lengths

### Memory Optimization
- Use quantization to reduce memory footprint
- Lower `gpu_memory_utilization` if experiencing OOM errors
- Consider using a smaller model if memory is constrained

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory (OOM) Errors**
   - Reduce `gpu_memory_utilization`
   - Decrease `max_model_len`
   - Use a smaller model or enable quantization

2. **Slow Performance**
   - Check GPU utilization with `nvidia-smi`
   - Ensure tensor cores are being used (check for mixed precision warnings)
   - Increase batch size if possible

3. **Model Loading Failures**
   - Verify model path/name is correct
   - Check available disk space in model cache directory
   - Ensure sufficient GPU memory is available

## FAQs

**Q: What models are supported?**
A: vLLM supports most popular open-source models including LLaMA, Mistral, and their variants.

**Q: Can I use vLLM with CPU-only?**
A: vLLM is optimized for GPU acceleration and doesn't support CPU-only operation.

**Q: How do I monitor GPU usage?**
A: Use `nvidia-smi` or the monitoring tools provided by your cloud provider.

**Q: Is there a maximum context length?**
A: The maximum context length is determined by `max_model_len` when loading the model.

## Additional Resources

- [vLLM Documentation](https://vllm.readthedocs.io/)
- [GitHub Repository](https://github.com/vllm-project/vllm)
- [Performance Benchmarks](https://vllm.ai/benchmarks/)
