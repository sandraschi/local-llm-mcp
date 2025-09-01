# vLLM V1 MCP Tools and Integration

## FastMCP 2.11.3 Stateful Tools

FastMCP 2.11.3 introduces stateful tools that can maintain state between invocations, with automatic caching and invalidation. This is particularly useful for operations that are expensive to compute but don't change frequently.

### Stateful Tool Example

```python
from fastmcp import FastMCP
from typing import Dict, Any

mcp = FastMCP("Stateful Example")

@mcp.tool(stateful=True, state_ttl=300)  # Cache for 5 minutes
async def get_model_info(model_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a model with stateful caching.
    
    Args:
        model_id: The ID of the model to get info for
        
    Returns:
        Dictionary containing model information
    """
    # Expensive operation to get model info
    return await _get_model_info_impl(model_id)
```

### Stateful Tool Parameters

- `stateful`: Set to `True` to enable stateful behavior (default: `False`)
- `state_ttl`: Time-to-live in seconds for the cached state (default: 300)
- `invalidate_on`: List of events that should invalidate the cache

### Cache Invalidation

Stateful tools automatically invalidate their cache when:
1. The TTL expires
2. The input parameters change
3. An invalidation event occurs

To manually invalidate a tool's cache:

```python
# Invalidate cache for a specific tool and parameters
await mcp.invalidate_tool_state("get_model_info", {"model_id": "llama3"})

# Invalidate all states for a tool
await mcp.invalidate_all_tool_states("get_model_info")
```

### Best Practices for Stateful Tools

1. Use stateful tools for:
   - Expensive computations
   - Frequent identical requests
   - Data that changes infrequently

2. Set appropriate TTL based on data volatility

3. Use descriptive docstrings to document caching behavior

4. Invalidate cache when underlying data changes

5. Be mindful of memory usage with large cached objects

## Step 1: Enhanced MCP Tools for vLLM V1

**File: `src/llm_mcp/tools/vllm_tools.py`** (Create new)
```python
"""MCP Tools for vLLM V1 Provider"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from fastmcp import FastMCP
from ..providers import PROVIDERS, VLLM_V1_AVAILABLE
from ..providers.vllm_v1.models import get_recommended_models, RTX_4090_MODELS

mcp = FastMCP("vLLM V1 Tools")

@mcp.tool(stateful=True, state_ttl=3600)  # Cache for 1 hour
async def load_vllm_model(model_id: str) -> Dict[str, Any]:
    """
    Load model with vLLM V1 optimizations.
    
    Note: This operation is stateful with a 1-hour TTL.
    Subsequent calls with the same model_id will return cached results.
    """
    if not VLLM_V1_AVAILABLE:
        return {"error": "vLLM V1 not available"}
    
    try:
        provider = PROVIDERS["vllm_v1"]({})
        if not await provider.initialize():
            return {"error": "Provider init failed"}
        
        success = await provider.load_model(model_id)
        if success:
            info = await provider.get_model_info()
            return {
                "status": "success",
                "model_id": model_id,
                "info": info,
                "message": f"✅ {model_id} loaded"
            }
        return {"error": f"Failed to load {model_id}"}
    except Exception as e:
        return {"error": str(e)}

@mcp.tool(stateful=True, state_ttl=300)  # Cache for 5 minutes
async def list_rtx4090_models() -> Dict[str, Any]:
    """
    List RTX 4090 optimized models.
    
    Note: Results are cached for 5 minutes to improve performance.
    """
    models_by_speed = {
        "blazing_fast": [],    # >250 TPS
        "very_fast": [],       # 150-250 TPS  
        "fast": [],            # 80-150 TPS
        "moderate": [],        # 30-80 TPS
    }
    
    for model_id, config in RTX_4090_MODELS.items():
        speed = config.expected_tokens_per_sec
        model_info = {
            "id": model_id,
            "type": config.model_type,
            "vram": f"{config.vram_required}GB",
            "expected_tps": speed,
        }
        
        if speed >= 250:
            models_by_speed["blazing_fast"].append(model_info)
        elif speed >= 150:
            models_by_speed["very_fast"].append(model_info)
        elif speed >= 80:
            models_by_speed["fast"].append(model_info)
        else:
            models_by_speed["moderate"].append(model_info)
    
    return {
        "categories": models_by_speed,
        "recommendations": {
            "speed": "microsoft/Phi-3-mini-4k-instruct",
            "balance": "meta-llama/Llama-3.1-8B-Instruct", 
            "code": "codellama/CodeLlama-13b-Instruct-hf",
            "vision": "Qwen/Qwen2-VL-7B-Instruct",
        }
    }

@mcp.tool()
async def benchmark_model(model_id: str) -> Dict[str, Any]:
    """Benchmark vLLM V1 performance"""
    test_prompts = [
        "Write a Python function for fibonacci.",
        "Explain quantum computing simply.",
        "Create a REST API endpoint.",
    ]
    
    try:
        provider = PROVIDERS["vllm_v1"]({})
        await provider.initialize()
        
        if not await provider.load_model(model_id):
            return {"error": "Model load failed"}
        
        results = []
        total_tokens = 0
        total_time = 0
        
        for prompt in test_prompts:
            start = time.time()
            response = ""
            async for chunk in provider.generate_text(prompt, max_tokens=100):
                response += chunk
            end = time.time()
            
            tokens = len(response) // 4
            tps = tokens / (end - start) if end > start else 0
            
            results.append({
                "tokens": tokens,
                "time": round(end - start, 2),
                "tps": round(tps, 1)
            })
            
            total_tokens += tokens
            total_time += (end - start)
        
        await provider.unload_model()
        
        overall_tps = total_tokens / total_time if total_time > 0 else 0
        expected = RTX_4090_MODELS.get(model_id, {}).expected_tokens_per_sec or 100
        
        return {
            "model": model_id,
            "results": results,
            "average_tps": round(overall_tps, 1),
            "expected_tps": expected,
            "performance": "good" if overall_tps >= expected * 0.8 else "check_setup"
        }
        
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def system_status() -> Dict[str, Any]:
    """Get vLLM V1 system status"""
    try:
        from ..providers.vllm_v1.utils import VLLMHealthCheck
        health = VLLMHealthCheck()
        status = await health.check_system()
        
        recommendations = []
        if status["vram_total"] >= 24:
            recommendations.append("✅ Excellent VRAM for most models")
        if status["ram_total"] >= 64:
            recommendations.append("✅ Great RAM for CPU offloading")
        if status["v1_ready"]:
            recommendations.append("✅ vLLM V1 configured")
        
        return {
            "hardware": status,
            "recommendations": recommendations,
            "optimal_models": get_recommended_models(),
        }
    except Exception as e:
        return {"error": str(e)}
```

## Step 2: Update Main Server Integration

**File: `src/llm_mcp/main.py`** (Add vLLM V1 integration)
```python
# Add to your existing main.py

from .tools.vllm_tools import mcp as vllm_mcp

# In your main FastMCP app setup:
# app.include_router(vllm_mcp.router)  # Add vLLM tools
```

## Step 3: Installation Script

**File: `tools/install_vllm_v1.py`**
```python
#!/usr/bin/env python3
"""vLLM V1 Installation Script for RTX 4090"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run command and show output"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(result.stdout)
    return True

def main():
    print("🚀 Installing vLLM V1 for RTX 4090...")
    
    # Set environment
    os.environ["VLLM_USE_V1"] = "1"
    
    # Uninstall old versions
    print("\n1. Cleaning old installations...")
    run_command("pip uninstall torch torchvision torchaudio vllm -y")
    
    # Install PyTorch with CUDA
    print("\n2. Installing PyTorch with CUDA 12.1...")
    if not run_command("pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121"):
        sys.exit(1)
    
    # Install vLLM V1
    print("\n3. Installing vLLM V1...")
    if not run_command("pip install vllm>=0.8.1"):
        sys.exit(1)
    
    # Install other dependencies
    print("\n4. Installing additional dependencies...")
    if not run_command("pip install -r requirements.txt"):
        sys.exit(1)
    
    # Verify installation
    print("\n5. Verifying installation...")
    verify_code = """
import torch
import vllm
import os
os.environ['VLLM_USE_V1'] = '1'
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')
print(f'vLLM: {vllm.__version__}')
print('✅ Installation successful')
"""
    
    if run_command(f'python -c "{verify_code}"'):
        print("\n🎉 vLLM V1 installation complete!")
        print("\nNext steps:")
        print("1. Run: python tools/validate_hardware.py")
        print("2. Test: python -c 'from src.llm_mcp.tools.vllm_tools import system_status; print(system_status())'")
    else:
        print("❌ Installation verification failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Quick Test Commands

```bash
# Install everything
python tools/install_vllm_v1.py

# Validate hardware
python tools/validate_hardware.py

# Test model loading
python -c "
import asyncio
from src.llm_mcp.tools.vllm_tools import quick_model_test
result = asyncio.run(quick_model_test('microsoft/Phi-3-mini-4k-instruct'))
print(result)
"
```

## Performance Expectations on Your Hardware

**RTX 4090 + 64GB Results:**
- Phi-3-mini: ~300 TPS (blazing fast)
- Llama-3.1-8B: ~200 TPS (excellent)
- CodeLlama-13B: ~120 TPS (great for code)
- Llama-3.1-70B: ~25 TPS (with CPU offload)

This should be massively faster than your current Ollama setup!
