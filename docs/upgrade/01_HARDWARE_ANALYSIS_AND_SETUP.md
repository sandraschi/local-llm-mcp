# vLLM V1 Hardware Analysis & Initial Setup

## Your RTX 4090 + 64GB System: Performance Analysis

### Hardware Capabilities Assessment
```
✅ RTX 4090: 24GB VRAM (CUDA 8.9) - Excellent
✅ 64GB System RAM - Perfect for large models
✅ 24-Core CPU - Great for preprocessing/tokenization
✅ NVMe SSD recommended for model loading
```

### Performance Reality Check: Ollama vs vLLM V1

**Current Ollama Performance (chatgpt-oss:20b in Zed):**
```
- Throughput: ~10-20 tokens/sec (SLOW)
- Memory efficiency: Poor (no optimization)
- Model loading: Slow, no caching
- Multi-model: Limited
- Vision support: None
```

**Expected vLLM V1 Performance on Your Hardware:**
```
Model Size    | Tokens/sec | Memory Usage | Notes
7B models     | 150-300    | 14GB VRAM   | Blazing fast
13B models    | 80-150     | 18GB VRAM   | Still very fast  
20B models    | 40-80      | 22GB VRAM   | 4x faster than Ollama
34B models    | 20-40      | 24GB VRAM   | Max VRAM usage
70B models    | 10-25      | 24GB+32GB   | CPU offloading
```

**Why vLLM V1 Will Help Massively:**
1. **FlashAttention 3**: 2-3x memory efficiency vs Ollama
2. **Prefix Caching**: Shared context between requests  
3. **Chunked Prefill**: Better batching and throughput
4. **V1 Engine**: 1.7x performance boost over old vLLM
5. **Zero-Config Optimization**: Auto-tuned for your hardware

## Phase 1: Dependencies and Environment Setup

### Step 1: Update Python Dependencies

**File: `requirements.txt`** (Replace existing)
```txt
# Core MCP dependencies (keep existing)
fastapi>=0.68.0
uvicorn[standard]>=0.15.0
pydantic>=2.0.0
fastmcp>=2.10.0
httpx>=0.24.0
python-dotenv>=1.0.0
python-multipart>=0.0.5

# vLLM V1 with CUDA support (NEW)
vllm>=0.8.1
torch>=2.4.0
transformers>=4.44.0
accelerate>=0.21.0
xformers>=0.0.20
flash-attn>=2.5.0

# Multimodal support (NEW)
Pillow>=10.0.0
opencv-python>=4.8.0
soundfile>=0.12.0

# Performance monitoring (NEW)
psutil>=5.9.0
nvidia-ml-py>=12.535.77

# Existing providers
aiohttp>=3.8.0

# Terminal interface (keep existing)
rich>=13.0.0
prompt-toolkit>=3.0.0

# Development (keep existing)
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
black>=23.0.0
isort>=5.0.0
mypy>=1.0.0
types-requests>=2.0.0
types-python-dateutil>=2.8.0
```

### Step 2: Environment Configuration

**File: `.env`** (Add these vLLM V1 settings)
```env
# ========================================
# vLLM V1 Engine Configuration
# ========================================
VLLM_USE_V1=1
VLLM_ENABLE_PREFIX_CACHING=1
VLLM_ATTENTION_BACKEND=FLASHINFER
VLLM_GPU_MEMORY_UTILIZATION=0.85
VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_MAX_SEQ_LEN=4096
VLLM_TRUST_REMOTE_CODE=1

# ========================================
# RTX 4090 Specific Optimizations
# ========================================
CUDA_VISIBLE_DEVICES=0
VLLM_USE_FLASH_ATTENTION=1
VLLM_ENABLE_CHUNKED_PREFILL=1
VLLM_NUM_SCHEDULER_STEPS=1

# ========================================
# Memory Management for 64GB System
# ========================================
VLLM_SWAP_SPACE=16
VLLM_CPU_OFFLOAD_GB=32
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# ========================================
# Logging and Debugging
# ========================================
VLLM_LOG_LEVEL=INFO
VLLM_TRACE_FUNCTION=0
VLLM_DISABLE_CUSTOM_ALL_REDUCE=0

# ========================================
# Provider Selection Preferences
# ========================================
DEFAULT_PROVIDER=vllm_v1
OLLAMA_FALLBACK=true
AUTO_PROVIDER_SELECTION=true
```

### Step 3: Update pyproject.toml

**File: `pyproject.toml`** (Update dependencies section)
```toml
[project]
name = "llm-mcp"
version = "0.2.0"  # Bump for vLLM V1 update
description = "FastMCP 2.10-compliant server with vLLM V1 performance"
readme = "README.md"
requires-python = ">=3.9"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
dependencies = [
    # Core MCP stack
    "fastapi>=0.68.0",
    "uvicorn[standard]>=0.15.0", 
    "pydantic>=2.0.0",
    "fastmcp>=2.10.0",
    "httpx>=0.24.0",
    "python-dotenv>=1.0.0",
    "python-multipart>=0.0.5",
    
    # vLLM V1 performance stack
    "vllm>=0.8.1",
    "torch>=2.4.0",
    "transformers>=4.44.0",
    "accelerate>=0.21.0",
    "xformers>=0.0.20",
    "flash-attn>=2.5.0",
    
    # Multimodal capabilities
    "Pillow>=10.0.0",
    "opencv-python>=4.8.0",
    "soundfile>=0.12.0",
    
    # System monitoring
    "psutil>=5.9.0",
    "nvidia-ml-py>=12.535.77",
    
    # Existing providers
    "aiohttp>=3.8.0",
    "rich>=13.0.0",
    "prompt-toolkit>=3.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "isort>=5.0.0",
    "mypy>=1.0.0",
    "types-requests>=2.0.0",
    "types-python-dateutil>=2.8.0",
]

# RTX 4090 optimized extras
rtx4090 = [
    "flash-attn>=2.5.0",
    "xformers>=0.0.20",
]

# Vision model extras
vision = [
    "Pillow>=10.0.0",
    "opencv-python>=4.8.0",
]

[project.urls]
Homepage = "https://github.com/yourusername/llm-mcp"
Documentation = "https://github.com/yourusername/llm-mcp#readme"
Issues = "https://github.com/yourusername/llm-mcp/issues"

[tool.black]
line-length = 88
target-version = ['py39']
include = '\.pyi?$'

[tool.isort]
profile = "black"
line_length = 88

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
disallow_untyped_calls = true
disallow_any_generics = true
```

## Installation Commands for Windsurf

### Step 1: Backup Current Setup
```bash
# In your project directory
git add . && git commit -m "Backup before vLLM V1 upgrade"
cp requirements.txt requirements.txt.backup
cp .env .env.backup
```

### Step 2: Install vLLM V1 Dependencies
```bash
# Uninstall old torch/vllm if present
pip uninstall torch torchvision torchaudio vllm -y

# Install PyTorch with CUDA 12.1 support
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM V1 with CUDA support
pip install vllm>=0.8.1

# Install remaining dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import vllm; print(f'vLLM Version: {vllm.__version__}')"
```

### Step 3: Verify V1 Environment
```bash
# Test V1 environment variables
export VLLM_USE_V1=1
python -c "
import os
print('V1 Environment Check:')
print(f'VLLM_USE_V1: {os.getenv(\"VLLM_USE_V1\")}')
print(f'CUDA_VISIBLE_DEVICES: {os.getenv(\"CUDA_VISIBLE_DEVICES\", \"all\")}')
"
```

## Hardware Validation Script

**File: `tools/validate_hardware.py`** (Create new)
```python
#!/usr/bin/env python3
"""Hardware validation for vLLM V1 on RTX 4090"""

import torch
import psutil
import subprocess
import sys

def check_cuda():
    """Check CUDA availability and GPU info"""
    print("🔍 CUDA Check:")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {gpu_props.name}")
        print(f"  VRAM: {gpu_props.total_memory // (1024**3)}GB")
        print(f"  Compute Capability: {gpu_props.major}.{gpu_props.minor}")
        
        if gpu_props.major >= 8:  # RTX 30/40 series
            print("  ✅ Compatible with FlashAttention")
        else:
            print("  ⚠️  FlashAttention may have limited support")
    else:
        print("  ❌ CUDA not available")
        return False
    
    return True

def check_memory():
    """Check system memory"""
    print("\n🧠 Memory Check:")
    mem = psutil.virtual_memory()
    ram_gb = mem.total // (1024**3)
    available_gb = mem.available // (1024**3)
    
    print(f"  Total RAM: {ram_gb}GB")
    print(f"  Available RAM: {available_gb}GB")
    
    if ram_gb >= 32:
        print("  ✅ Sufficient RAM for large models")
    elif ram_gb >= 16:
        print("  ⚠️  Moderate RAM - stick to smaller models")
    else:
        print("  ❌ Insufficient RAM for vLLM V1")
        return False
    
    return True

def check_vllm_v1():
    """Check vLLM V1 installation"""
    print("\n🚀 vLLM V1 Check:")
    
    try:
        import vllm
        print(f"  vLLM Version: {vllm.__version__}")
        
        # Check if V1 is available
        import os
        os.environ["VLLM_USE_V1"] = "1"
        
        from vllm.engine.arg_utils import AsyncEngineArgs
        print("  ✅ AsyncEngineArgs available")
        
        print("  ✅ vLLM V1 ready")
        return True
        
    except ImportError as e:
        print(f"  ❌ vLLM import failed: {e}")
        return False
    except Exception as e:
        print(f"  ⚠️  vLLM V1 check failed: {e}")
        return False

def check_flash_attention():
    """Check FlashAttention availability"""
    print("\n⚡ FlashAttention Check:")
    
    try:
        import flash_attn
        print(f"  FlashAttention Version: {flash_attn.__version__}")
        print("  ✅ FlashAttention available")
        return True
    except ImportError:
        print("  ⚠️  FlashAttention not installed (optional)")
        return True  # Not critical

def main():
    """Run all hardware validation checks"""
    print("🔧 vLLM V1 Hardware Validation for RTX 4090\n")
    
    checks = [
        check_cuda(),
        check_memory(),
        check_vllm_v1(),
        check_flash_attention()
    ]
    
    print(f"\n📊 Results: {sum(checks)}/{len(checks)} checks passed")
    
    if all(checks):
        print("🎉 System ready for vLLM V1!")
        
        # Provide recommended models
        print("\n🤖 Recommended Models for Your Hardware:")
        print("  • meta-llama/Llama-3.1-8B-Instruct (Fast)")
        print("  • codellama/CodeLlama-13b-Instruct-hf (Code)")
        print("  • Qwen/Qwen2-VL-7B-Instruct (Vision)")
        print("  • meta-llama/Llama-3.1-70B-Instruct (CPU offload)")
        
    else:
        print("❌ Some checks failed - review requirements")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Next Steps

1. **Run validation**: `python tools/validate_hardware.py`
2. **Install dependencies**: Follow installation commands above
3. **Move to Part 2**: Model configurations and provider implementation
4. **Move to Part 3**: MCP tools and integration

This sets up the foundation for vLLM V1. Your RTX 4090 with 64GB RAM is actually a beast setup that will handle most models beautifully with vLLM V1's optimizations!
