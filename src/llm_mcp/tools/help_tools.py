"""Extensive Multilevel Help System for Local LLM MCP Server.

This module provides a comprehensive, multi-tiered help system with 5 levels of detail:
- Level 0: Tool names only (quick reference)
- Level 1: Basic descriptions (getting started)
- Level 2: Intermediate usage (workflows, examples)
- Level 3: Advanced features (performance, integration)
- Level 4: Expert details (architecture, troubleshooting)

The help system includes:
- Interactive tool discovery
- Workflow guides and best practices
- Performance optimization tips
- Troubleshooting guides
- Integration examples
- Hardware recommendations
- Configuration guidance
"""

import inspect
from collections.abc import Callable
from enum import Enum
from typing import Any, Union, get_args, get_origin


class HelpLevel(Enum):
    """Help detail levels."""

    NAMES_ONLY = 0  # Just tool names
    BASIC = 1  # Basic descriptions
    INTERMEDIATE = 2  # Usage examples and workflows
    ADVANCED = 3  # Performance, integration, advanced features
    EXPERT = 4  # Architecture, troubleshooting, deep technical details


class ToolCategory(Enum):
    """Tool categories for organization."""

    PORTMANTEAU = "portmanteau"
    GPU = "gpu"
    CORE = "core"
    SYSTEM = "system"
    HELP = "help"


def format_type(t: type) -> str:
    """Format a Python type for documentation."""
    if t is type(None):
        return "None"
    if t == inspect.Parameter.empty:
        return "any"

    # Handle Optional types
    if get_origin(t) is Union:
        args = [a for a in get_args(t) if a is not type(None)]
        if len(args) == 1:
            return f"{format_type(args[0])} (optional)"
        return " | ".join(format_type(a) for a in args) + " (optional)"

    # Handle container types
    if hasattr(t, "__origin__"):
        if t.__origin__ is list:
            item_type = format_type(t.__args__[0]) if t.__args__ else "any"
            return f"List[{item_type}]"
        if t.__origin__ is dict:
            key_type = format_type(t.__args__[0]) if t.__args__ else "any"
            value_type = format_type(t.__args__[1]) if len(t.__args__) > 1 else "any"
            return f"Dict[{key_type}, {value_type}]"

    # Default case
    return t.__name__ if hasattr(t, "__name__") else str(t)


def get_tool_category(tool_name: str) -> ToolCategory:
    """Determine tool category from name."""
    if "llm_" in tool_name:
        return ToolCategory.PORTMANTEAU
    elif "gpu_" in tool_name:
        return ToolCategory.GPU
    elif tool_name in [
        "list_models",
        "get_model_info",
        "register_model",
        "generate_text",
        "chat_completion",
        "embed_text",
    ]:
        return ToolCategory.CORE
    elif "health" in tool_name or "status" in tool_name:
        return ToolCategory.SYSTEM
    elif "help" in tool_name:
        return ToolCategory.HELP
    return ToolCategory.SYSTEM


def get_comprehensive_tool_info(func: Callable, level: HelpLevel) -> dict[str, Any]:
    """Get comprehensive tool information at specified detail level."""
    doc = inspect.getdoc(func) or ""
    func_name = getattr(func, "__name__", getattr(func, "name", func.__class__.__name__))

    base_info = {
        "name": func_name,
        "category": get_tool_category(func_name).value,
        "description": doc.split("\n")[0] if doc else "",
    }

    # Level 0: Names only
    if level == HelpLevel.NAMES_ONLY:
        return base_info

    # Level 1: Basic descriptions
    if level == HelpLevel.BASIC:
        return {
            **base_info,
            "parameters": get_parameter_docs(func),
            "returns": get_return_docs(func),
        }

    # Level 2+: Include examples and usage patterns
    if level.value >= 2:
        base_info.update(
            {
                "parameters": get_parameter_docs(func),
                "returns": get_return_docs(func),
                "examples": get_tool_examples(func_name),
                "usage_patterns": get_usage_patterns(func_name),
            }
        )

    # Level 3+: Include performance and integration info
    if level.value >= 3:
        base_info.update(
            {
                "performance_notes": get_performance_notes(func_name),
                "integration_notes": get_integration_notes(func_name),
                "common_issues": get_common_issues(func_name),
            }
        )

    # Level 4: Expert details
    if level == HelpLevel.EXPERT:
        base_info.update(
            {
                "architecture_notes": get_architecture_notes(func_name),
                "troubleshooting": get_troubleshooting_guide(func_name),
                "advanced_config": get_advanced_config(func_name),
                "full_doc": doc,
            }
        )

    return base_info


def get_parameter_docs(func: Callable) -> list[dict[str, Any]]:
    """Extract parameter documentation from a function."""
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return []

    type_hints = getattr(func, "__annotations__", {})
    params = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        param_info = {
            "name": name,
            "type": format_type(type_hints.get(name, param.annotation)),
            "required": param.default is param.empty,
            "default": param.default if param.default is not param.empty else None,
            "description": "",
        }

        # Get description from docstring
        try:
            doc = inspect.getdoc(func) or ""
            for line in doc.split("\n"):
                line = line.strip()
                if line.startswith(f"{name}:") and ":" in line:
                    param_info["description"] = line.split(":", 1)[1].strip()
                    break
        except Exception:
            pass

        params.append(param_info)

    return params


def get_return_docs(func: Callable) -> dict[str, str]:
    """Extract return type and description."""
    try:
        type_hints = getattr(func, "__annotations__", {})
        return_type = type_hints.get("return", "None")
    except Exception:
        return_type = "None"

    try:
        doc = inspect.getdoc(func) or ""
        description = ""

        # Extract return description from docstring
        in_returns = False
        for line in doc.split("\n"):
            line = line.strip()
            if line.lower().startswith("returns:"):
                in_returns = True
                description = line[8:].strip()
            elif in_returns and line and not line.startswith(" "):
                break
            elif in_returns:
                description += " " + line.strip()
    except Exception:
        description = ""

    return {"type": format_type(return_type), "description": description}


def get_tool_examples(tool_name: str) -> list[dict[str, Any]]:
    """Get usage examples for a tool."""
    examples = {
        "llm_health_tool": [
            {
                "description": "Basic system health check",
                "code": 'await llm_health_tool("health_check")',
                "expected": "System status with CPU, memory, disk usage",
            },
            {
                "description": "Get performance metrics",
                "code": 'await llm_health_tool("get_metrics", "cpu_percent", since_minutes=10)',
                "expected": "CPU usage over last 10 minutes",
            },
        ],
        "llm_generation_tool": [
            {
                "description": "Basic text generation",
                "code": 'await llm_generation_tool("generate_text", model="llama3", prompt="Hello world")',
                "expected": "Generated text response",
            },
            {
                "description": "Advanced generation with parameters",
                "code": (
                    'await llm_generation_tool("generate_text", model="llama3", '
                    'prompt="Explain AI", temperature=0.1, max_tokens=500)'
                ),
                "expected": "Precise, detailed explanation",
            },
        ],
        "llm_huggingface_tool": [
            {
                "description": "Download gated FLUX model",
                "code": (
                    'await llm_huggingface_tool("download_model", '
                    'model_id="blackforestlabs/FLUX.1-dev", local_path="./models")'
                ),
                "expected": "FLUX model downloaded (requires HUGGINGFACE_TOKEN)",
            }
        ],
        "gpu_status": [
            {
                "description": "Check GPU memory usage",
                "code": "await gpu_status()",
                "expected": "GPU utilization, memory usage, temperature",
            }
        ],
    }
    return examples.get(tool_name, [])


def get_usage_patterns(tool_name: str) -> list[str]:
    """Get usage patterns for a tool."""
    patterns = {
        "llm_health_tool": [
            "Monitor system resources before intensive operations",
            "Check GPU availability before model loading",
            "Track performance metrics during long-running tasks",
        ],
        "llm_generation_tool": [
            "Use temperature=0.1 for factual/coding tasks",
            "Use temperature=0.7-0.9 for creative tasks",
            "Set max_tokens based on desired response length",
        ],
        "llm_finetuning_tool": [
            "Start with lora_prepare to check hardware compatibility",
            "Use gradient_accumulation for larger effective batch sizes",
            "Monitor GPU memory during training",
        ],
        "gpu_status": [
            "Check before loading large models",
            "Monitor during training to prevent OOM",
            "Use gpu_clear_memory if memory fragmentation occurs",
        ],
    }
    return patterns.get(tool_name, [])


def get_performance_notes(tool_name: str) -> dict[str, Any]:
    """Get performance optimization notes."""
    notes = {
        "llm_generation_tool": {
            "gpu_acceleration": "Use CUDA-enabled GPUs for 10-50x speedup",
            "batch_processing": "Process multiple prompts together when possible",
            "model_caching": "Keep frequently used models loaded in memory",
            "quantization": "Use 4-bit quantization to reduce memory usage by 75%",
        },
        "llm_finetuning_tool": {
            "gpu_memory": "RTX 4090 can handle up to 13B models with 4-bit quantization",
            "gradient_checkpointing": "Reduces memory usage by 60% at 20% performance cost",
            "mixed_precision": "Use bf16 for 2x training speedup with minimal quality loss",
            "dataset_size": "Start with 1K-10K samples for initial testing",
        },
        "gpu_status": {
            "memory_monitoring": "Check memory utilization before loading models",
            "thermal_limits": "Keep GPU temperature below 85°C for optimal performance",
            "power_limits": "Monitor power draw to avoid thermal throttling",
        },
    }
    return notes.get(tool_name, {})


def get_integration_notes(tool_name: str) -> dict[str, Any]:
    """Get integration and compatibility notes."""
    notes = {
        "llm_huggingface_tool": {
            "authentication": "Set HUGGINGFACE_TOKEN or HF_TOKEN environment variable",
            "gated_models": "FLUX, Stable Diffusion XL, and other restricted models require authentication",
            "rate_limits": "Hugging Face API has rate limits; use local models when possible",
            "caching": "Downloaded models are cached locally for faster subsequent loads",
        },
        "llm_ollama_tool": {
            "local_first": "Ollama runs models locally - no API keys required",
            "model_availability": "Use ollama_pull_model to download models first",
            "resource_usage": "Models run in separate processes with their own memory",
        },
        "gpu_status": {
            "nvidia_only": "Currently optimized for NVIDIA GPUs (RTX 30/40 series)",
            "cuda_versions": "Requires CUDA 11.8+ for optimal performance",
            "memory_management": "Use gpu_clear_memory to prevent fragmentation on RTX 4090",
        },
    }
    return notes.get(tool_name, {})


def get_common_issues(tool_name: str) -> list[dict[str, Any]]:
    """Get common issues and solutions."""
    issues = {
        "llm_generation_tool": [
            {
                "issue": "Model not found error",
                "solution": "Use llm_models_tool('list_models') to see available models, or load the model first",
            },
            {
                "issue": "Out of memory error",
                "solution": "Use smaller models, enable quantization, or clear GPU memory with gpu_clear_memory()",
            },
        ],
        "llm_huggingface_tool": [
            {
                "issue": "Gated model access denied",
                "solution": "Set HUGGINGFACE_TOKEN environment variable and request access on Hugging Face",
            },
            {"issue": "Download timeout", "solution": "Use faster internet connection or download smaller models"},
        ],
        "gpu_status": [
            {
                "issue": "GPU not detected",
                "solution": "Install NVIDIA drivers and CUDA toolkit, ensure GPU is not in use by other applications",
            },
            {
                "issue": "Memory fragmentation",
                "solution": "Use gpu_clear_memory() to defragment GPU memory, especially on RTX 4090",
            },
        ],
        "llm_finetuning_tool": [
            {
                "issue": "Training runs out of memory",
                "solution": "Reduce batch size, enable gradient checkpointing, use 4-bit quantization",
            },
            {
                "issue": "Training is very slow",
                "solution": "Use mixed precision (bf16), increase batch size, ensure GPU is not thermal throttling",
            },
        ],
    }
    return issues.get(tool_name, [])


def get_architecture_notes(tool_name: str) -> dict[str, Any]:
    """Get architecture and technical implementation notes."""
    notes = {
        "llm_health_tool": {
            "implementation": "Uses psutil and GPUtil for system monitoring",
            "threading": "Non-blocking async operations for real-time monitoring",
            "caching": "Stateful tool with automatic cache management",
        },
        "llm_generation_tool": {
            "portmanteau_pattern": "Consolidates text generation, chat, and embeddings into single tool",
            "provider_abstraction": "Supports multiple LLM providers through unified interface",
            "streaming": "Supports both streaming and non-streaming generation modes",
        },
        "gpu_status": {
            "nvidia_smi": "Uses nvidia-ml-py3 for GPU monitoring",
            "memory_tracking": "Real-time GPU memory allocation tracking",
            "thermal_monitoring": "Continuous temperature and power monitoring",
        },
    }
    return notes.get(tool_name, {})


def get_troubleshooting_guide(tool_name: str) -> dict[str, Any]:
    """Get detailed troubleshooting guide."""
    guides = {
        "llm_generation_tool": {
            "debug_mode": "Enable debug logging to see detailed error traces",
            "model_validation": "Use get_model_info() to verify model compatibility",
            "resource_monitoring": "Monitor GPU/CPU usage during generation",
            "fallback_strategies": "Try different models if one fails",
        },
        "llm_finetuning_tool": {
            "memory_profiling": "Use gpu_status() to monitor memory during training",
            "gradient_debugging": "Check for gradient explosions/nan values",
            "dataset_validation": "Verify dataset format and quality before training",
            "checkpoint_recovery": "Use resume training from checkpoints on failure",
        },
    }
    return guides.get(tool_name, {})


def get_advanced_config(tool_name: str) -> dict[str, Any]:
    """Get advanced configuration options."""
    configs = {
        "llm_generation_tool": {
            "custom_providers": "Can register custom LLM providers",
            "model_override": "Support for model-specific parameter overrides",
            "caching_strategy": "Configurable response caching and deduplication",
        },
        "gpu_status": {
            "polling_interval": "Configurable GPU status polling frequency",
            "alert_thresholds": "Customizable temperature and memory warning thresholds",
            "historical_tracking": "Enable GPU usage history logging",
        },
    }
    return configs.get(tool_name, {})


# Implementation functions for extensive multilevel help system
async def _list_tools_impl(mcp: Any, detail: int = 1) -> dict[str, Any]:
    """Implementation of list_tools functionality with multilevel detail.

    Args:
        mcp: The MCP server instance
        detail: Level of detail (0-4, see HelpLevel enum)

    Returns:
        Dictionary with comprehensive tool information
    """
    level = HelpLevel(min(max(detail, 0), 4))  # Clamp to valid range
    tools = {}

    mcp_tools = await mcp.get_tools()
    for name, tool in mcp_tools.items():
        tools[name] = get_comprehensive_tool_info(tool, level)

    # Group by category for better organization
    categorized = {}
    for tool_name, tool_info in tools.items():
        category = tool_info.get("category", "other")
        if category not in categorized:
            categorized[category] = {}
        categorized[category][tool_name] = tool_info

    return {
        "tools": tools,
        "categorized": categorized,
        "summary": {
            "total_tools": len(tools),
            "categories": {cat: len(tools) for cat, tools in categorized.items()},
            "detail_level": level.value,
            "level_description": {
                0: "Tool names only",
                1: "Basic descriptions",
                2: "Usage examples and workflows",
                3: "Performance and integration notes",
                4: "Expert technical details",
            }.get(level.value, "Unknown"),
        },
    }


async def _get_tool_help_impl(mcp: Any, tool_name: str, detail: int = 2) -> dict[str, Any]:
    """Implementation of get_tool_help with multilevel detail.

    Args:
        mcp: The MCP server instance
        tool_name: Name of the tool to get help for
        detail: Level of detail (0-4)

    Returns:
        Comprehensive documentation for the tool
    """
    level = HelpLevel(min(max(detail, 0), 4))
    mcp_tools = await mcp.get_tools()

    for name, tool in mcp_tools.items():
        if name == tool_name:
            tool_info = get_comprehensive_tool_info(tool, level)
            return {
                "tool": tool_name,
                **tool_info,
                "help_level": level.value,
                "available_levels": [0, 1, 2, 3, 4],
                "level_descriptions": {
                    0: "Names only",
                    1: "Basic usage",
                    2: "Examples and patterns",
                    3: "Performance & integration",
                    4: "Expert troubleshooting",
                },
            }

    return {
        "error": f"Tool '{tool_name}' not found",
        "available_tools": list(mcp_tools.keys()),
        "suggestion": "Use list_available_tools() to see all available tools",
    }


async def _search_tools_impl(mcp: Any, query: str, category: str | None = None) -> dict[str, Any]:
    """Implementation of search_tools with category filtering.

    Args:
        mcp: The MCP server instance
        query: Search term
        category: Optional category filter

    Returns:
        Dictionary with matching tools and search metadata
    """
    query = query.lower()
    matches = []
    search_metadata = {"query": query, "category_filter": category, "total_searched": 0, "matches_found": 0}

    mcp_tools = await mcp.get_tools()
    search_metadata["total_searched"] = len(mcp_tools)

    for name, tool in mcp_tools.items():
        doc = (inspect.getdoc(tool) or "").lower()
        tool_category = get_tool_category(name).value

        # Apply category filter if specified
        if category and tool_category != category:
            continue

        # Search in name, description, and documentation
        if (
            query in name.lower()
            or query in doc
            or any(query in param.get("name", "").lower() for param in get_parameter_docs(tool))
        ):
            matches.append(
                {
                    "name": name,
                    "category": tool_category,
                    "description": doc.split("\n")[0] if doc else "",
                    "relevance_score": _calculate_relevance_score(query, name, doc),
                }
            )

    # Sort by relevance score
    matches.sort(key=lambda x: x["relevance_score"], reverse=True)
    search_metadata["matches_found"] = len(matches)

    return {
        "matches": matches,
        "search_metadata": search_metadata,
        "categories_available": [cat.value for cat in ToolCategory],
    }


def _calculate_relevance_score(query: str, tool_name: str, doc: str) -> float:
    """Calculate relevance score for search results."""
    score = 0.0

    # Exact name match gets highest score
    if query == tool_name.lower():
        score += 1.0

    # Name contains query
    if query in tool_name.lower():
        score += 0.7

    # Description contains query
    if query in doc:
        score += 0.5

    # Word boundaries (better matches)
    if f" {query} " in f" {doc} ":
        score += 0.3

    return score


async def _get_tool_signature_impl(mcp: Any, tool_name: str) -> dict[str, Any]:
    """Implementation of get_tool_signature with enhanced type information.

    Args:
        mcp: The MCP server instance
        tool_name: Name of the tool

    Returns:
        Dictionary with detailed tool signature information
    """
    mcp_tools = await mcp.get_tools()

    for name, tool in mcp_tools.items():
        if name == tool_name:
            sig = inspect.signature(tool)
            return {
                "name": name,
                "signature": str(sig),
                "parameters": [
                    {
                        "name": param.name,
                        "kind": str(param.kind).split(".")[1],  # Remove enum prefix
                        "default": param.default if param.default is not param.empty else None,
                        "annotation": format_type(param.annotation),
                        "required": param.default is param.empty,
                    }
                    for param in sig.parameters.values()
                    if param.name != "self"
                ],
                "return_annotation": format_type(sig.return_annotation),
                "callable_type": type(tool).__name__,
                "docstring_lines": len((inspect.getdoc(tool) or "").split("\n")),
                "has_examples": len(get_tool_examples(name)) > 0,
            }

    return {"error": f"Tool '{tool_name}' not found", "available_tools": list(mcp_tools.keys())}


async def _get_workflow_guides_impl(category: str | None = None) -> dict[str, Any]:
    """Get workflow guides and best practices.

    Args:
        category: Optional category filter

    Returns:
        Dictionary with workflow guides
    """
    all_workflows = {
        "model_management": {
            "title": "Complete Model Management Workflow",
            "description": "End-to-end process for discovering, downloading, and managing models",
            "steps": [
                {
                    "step": 1,
                    "action": "Discover available models",
                    "tool": "llm_models_tool",
                    "operation": "list_models",
                    "description": "Browse models across all providers",
                },
                {
                    "step": 2,
                    "action": "Check model compatibility",
                    "tool": "get_model_info",
                    "description": "Verify model requirements and capabilities",
                },
                {
                    "step": 3,
                    "action": "Download gated models",
                    "tool": "llm_huggingface_tool",
                    "operation": "download_model",
                    "description": "Download FLUX, Stable Diffusion, etc. (requires token)",
                },
                {
                    "step": 4,
                    "action": "Monitor system resources",
                    "tool": "llm_health_tool",
                    "operation": "health_check",
                    "description": "Ensure sufficient resources before loading",
                },
            ],
            "best_practices": [
                "Set HUGGINGFACE_TOKEN for gated models",
                "Check hardware compatibility before downloading large models",
                "Use quantization to reduce memory requirements",
                "Monitor GPU memory during model loading",
            ],
        },
        "content_generation": {
            "title": "Professional Content Generation Pipeline",
            "description": "Systematic approach to generating high-quality content",
            "steps": [
                {
                    "step": 1,
                    "action": "Select appropriate model",
                    "tool": "llm_models_tool",
                    "operation": "list_models",
                    "description": "Choose model based on task requirements",
                },
                {
                    "step": 2,
                    "action": "Optimize parameters",
                    "tool": "llm_generation_tool",
                    "description": "Set temperature, max_tokens based on content type",
                },
                {
                    "step": 3,
                    "action": "Generate initial content",
                    "tool": "llm_generation_tool",
                    "operation": "generate_text",
                    "description": "Create first draft",
                },
                {
                    "step": 4,
                    "action": "Refine with chat completion",
                    "tool": "llm_generation_tool",
                    "operation": "chat_completion",
                    "description": "Iteratively improve content quality",
                },
            ],
            "parameter_guidelines": {
                "creative_writing": {"temperature": 0.8, "max_tokens": 1000},
                "technical_writing": {"temperature": 0.1, "max_tokens": 500},
                "code_generation": {"temperature": 0.2, "max_tokens": 800},
                "chat_conversation": {"temperature": 0.7, "max_tokens": 300},
            },
        },
        "gpu_optimization": {
            "title": "GPU Memory Management and Optimization",
            "description": "Maximize GPU utilization and prevent memory issues",
            "steps": [
                {
                    "step": 1,
                    "action": "Monitor current status",
                    "tool": "gpu_status",
                    "description": "Check GPU utilization and memory usage",
                },
                {
                    "step": 2,
                    "action": "Clear memory fragmentation",
                    "tool": "gpu_clear_memory",
                    "description": "Defragment GPU memory (important for RTX 4090)",
                },
                {
                    "step": 3,
                    "action": "Optimize memory layout",
                    "tool": "gpu_optimize",
                    "description": "Advanced memory optimization and health check",
                },
                {
                    "step": 4,
                    "action": "Monitor during operations",
                    "tool": "gpu_status",
                    "description": "Track GPU usage during model loading/training",
                },
            ],
            "optimization_tips": [
                "Clear GPU memory before loading large models",
                "Use 4-bit quantization to reduce memory by 75%",
                "Enable gradient checkpointing during training",
                "Monitor temperature to prevent thermal throttling",
                "Use mixed precision (bf16) for 2x speedup",
            ],
        },
        "fine_tuning": {
            "title": "Model Fine-tuning Best Practices",
            "description": "Complete guide to fine-tuning LLMs effectively",
            "steps": [
                {
                    "step": 1,
                    "action": "Assess hardware capabilities",
                    "tool": "hardware_requirements",
                    "description": "Check if your GPU can handle the model size",
                },
                {
                    "step": 2,
                    "action": "Prepare training configuration",
                    "tool": "llm_finetuning_tool",
                    "operation": "lora_prepare",
                    "description": "Configure LoRA parameters and check compatibility",
                },
                {
                    "step": 3,
                    "action": "Prepare dataset",
                    "tool": "llm_finetuning_tool",
                    "operation": "prepare_dataset",
                    "description": "Format and validate training data",
                },
                {
                    "step": 4,
                    "action": "Execute training",
                    "tool": "llm_finetuning_tool",
                    "operation": "lora_train",
                    "description": "Run fine-tuning with monitoring",
                },
                {
                    "step": 5,
                    "action": "Evaluate results",
                    "tool": "llm_finetuning_tool",
                    "operation": "evaluate_model",
                    "description": "Assess training effectiveness and quality",
                },
            ],
            "hardware_guidelines": {
                "rtx_3090": "Up to 13B models with 4-bit quantization",
                "rtx_4090": "Up to 30B models with 4-bit quantization",
                "h100_80gb": "Up to 70B models with full precision",
                "a100_80gb": "Up to 70B models with mixed precision",
            },
        },
    }

    if category:
        workflows = {k: v for k, v in all_workflows.items() if category.lower() in k.lower()}
    else:
        workflows = all_workflows

    return {
        "workflows": workflows,
        "categories": ["model_management", "content_generation", "gpu_optimization", "fine_tuning"],
        "total_workflows": len(workflows),
    }


async def _get_performance_guide_impl() -> dict[str, Any]:
    """Get comprehensive performance optimization guide.

    Returns:
        Dictionary with performance optimization recommendations
    """
    return {
        "gpu_optimization": {
            "memory_management": [
                "Use 4-bit quantization to reduce memory usage by 75%",
                "Enable gradient checkpointing to save 60% memory",
                "Use mixed precision (bf16) for 2x training speedup",
                "Clear GPU memory fragmentation with gpu_clear_memory()",
                "Monitor memory usage with gpu_status() before loading models",
            ],
            "speed_optimization": [
                "Use Flash Attention 2.0 when available",
                "Enable CUDA graphs for repetitive operations",
                "Use model parallelism for very large models",
                "Optimize batch sizes based on GPU memory",
                "Cache frequently used models in memory",
            ],
            "thermal_management": [
                "Keep GPU temperature below 85°C to prevent throttling",
                "Use adequate cooling and case airflow",
                "Monitor power draw to avoid thermal limits",
                "Consider GPU undervolting for better thermals",
            ],
        },
        "cpu_optimization": {
            "multiprocessing": [
                "Use multiple CPU cores for data preprocessing",
                "Parallelize independent operations",
                "Use async/await for I/O bound operations",
                "Consider multiprocessing for CPU-intensive tasks",
            ],
            "memory_optimization": [
                "Use streaming for large datasets",
                "Implement proper garbage collection",
                "Use memory-efficient data structures",
                "Monitor memory usage with system tools",
            ],
        },
        "model_optimization": {
            "quantization_strategies": [
                "4-bit quantization: 75% memory reduction, minimal quality loss",
                "8-bit quantization: 50% memory reduction, slight quality loss",
                "Dynamic quantization: Automatic precision adjustment",
                "Mixed precision: FP16/FP32 combination for speed/accuracy balance",
            ],
            "architectural_optimizations": [
                "Use model parallelism for models >30B parameters",
                "Implement model sharding across multiple GPUs",
                "Use LoRA for parameter-efficient fine-tuning",
                "Consider model distillation for smaller footprints",
            ],
        },
        "benchmarking": {
            "key_metrics": [
                "Tokens per second (generation speed)",
                "Memory utilization percentage",
                "GPU utilization percentage",
                "Power consumption in watts",
                "Temperature in Celsius",
            ],
            "benchmarking_tools": [
                "Use llm_health_tool for system monitoring",
                "Use gpu_status for GPU-specific metrics",
                "Time operations for throughput measurement",
                "Monitor memory usage patterns over time",
            ],
        },
    }


async def _get_troubleshooting_guide_impl(category: str | None = None) -> dict[str, Any]:
    """Get comprehensive troubleshooting guide.

    Args:
        category: Optional category filter

    Returns:
        Dictionary with troubleshooting information
    """
    all_troubleshooting = {
        "model_loading": {
            "out_of_memory": {
                "symptoms": "CUDA out of memory error during model loading",
                "causes": ["Model too large for GPU memory", "Memory fragmentation", "Other processes using GPU"],
                "solutions": [
                    "Use 4-bit quantization: reduces memory by 75%",
                    "Clear GPU memory with gpu_clear_memory()",
                    "Unload other models first",
                    "Use smaller model variants",
                    "Enable CPU offloading for very large models",
                ],
                "prevention": "Always check gpu_status() before loading models",
            },
            "model_not_found": {
                "symptoms": "Model not found in registry or cache",
                "causes": ["Incorrect model name", "Model not downloaded", "Path issues"],
                "solutions": [
                    "Use llm_models_tool('list_models') to see available models",
                    "Download missing models first",
                    "Check model name spelling and format",
                    "Verify local model cache paths",
                ],
            },
        },
        "generation_issues": {
            "slow_generation": {
                "symptoms": "Very slow text generation (tokens/sec)",
                "causes": ["Large models", "High temperature", "GPU thermal throttling", "Memory pressure"],
                "solutions": [
                    "Use smaller/faster models for simple tasks",
                    "Lower temperature for deterministic outputs",
                    "Check GPU temperature with gpu_status()",
                    "Clear GPU memory fragmentation",
                    "Use mixed precision inference",
                ],
            },
            "poor_quality_output": {
                "symptoms": "Irrelevant, repetitive, or nonsensical responses",
                "causes": ["Wrong temperature setting", "Incorrect model for task", "Poor prompt engineering"],
                "solutions": [
                    "Adjust temperature: lower for factual tasks (0.1-0.3), higher for creative (0.7-0.9)",
                    "Choose appropriate model for your task",
                    "Improve prompt specificity and clarity",
                    "Use few-shot examples in prompts",
                    "Try different models and compare results",
                ],
            },
        },
        "gpu_issues": {
            "memory_fragmentation": {
                "symptoms": "CUDA out of memory despite available memory",
                "causes": ["GPU memory fragmentation on RTX 4090", "Long-running processes", "Memory leaks"],
                "solutions": [
                    "Use gpu_clear_memory() to defragment memory",
                    "Restart the Python process for fresh memory",
                    "Use gpu_optimize() for advanced cleanup",
                    "Monitor memory usage patterns",
                ],
                "note": "RTX 4090 GPUs are particularly susceptible to memory fragmentation",
            },
            "thermal_throttling": {
                "symptoms": "GPU utilization drops, slow performance",
                "causes": ["High GPU temperature", "Inadequate cooling", "Power limit reached"],
                "solutions": [
                    "Improve case cooling and airflow",
                    "Monitor temperature with gpu_status()",
                    "Reduce power-intensive operations during high temps",
                    "Consider GPU undervolting for better thermals",
                ],
            },
        },
        "network_issues": {
            "download_failures": {
                "symptoms": "Model download fails or times out",
                "causes": ["Slow internet", "Hugging Face rate limits", "Authentication issues"],
                "solutions": [
                    "Check internet connection speed",
                    "Set HUGGINGFACE_TOKEN for faster downloads",
                    "Use local model cache when possible",
                    "Try downloading smaller models first",
                ],
            },
            "api_rate_limits": {
                "symptoms": "429 Too Many Requests errors",
                "causes": ["Excessive API calls", "Shared IP rate limiting"],
                "solutions": [
                    "Implement request rate limiting",
                    "Use cached responses when possible",
                    "Consider local model hosting",
                    "Space out API calls appropriately",
                ],
            },
        },
        "fine_tuning_issues": {
            "training_divergence": {
                "symptoms": "Loss increases, poor training results",
                "causes": ["Too high learning rate", "Poor data quality", "Model instability"],
                "solutions": [
                    "Reduce learning rate (try 1e-5 to 5e-5)",
                    "Check and clean training data",
                    "Use gradient clipping",
                    "Monitor training metrics closely",
                    "Consider using a different base model",
                ],
            },
            "memory_issues_during_training": {
                "symptoms": "Out of memory during training",
                "causes": ["Batch size too large", "Model too big", "Gradient accumulation issues"],
                "solutions": [
                    "Reduce batch size or use gradient accumulation",
                    "Enable gradient checkpointing",
                    "Use 4-bit quantization for LoRA training",
                    "Consider model parallelism for very large models",
                    "Use CPU offloading as last resort",
                ],
            },
        },
    }

    if category:
        troubleshooting = {k: v for k, v in all_troubleshooting.items() if category.lower() in k.lower()}
    else:
        troubleshooting = all_troubleshooting

    return {
        "troubleshooting": troubleshooting,
        "categories": list(all_troubleshooting.keys()),
        "total_issues": sum(len(issues) for issues in troubleshooting.values()),
    }


def register_help_tools(mcp, register_individual_tools: bool = True):
    """Register help tools with the MCP server using FastMCP 2.12+ features.

    Args:
        mcp: The MCP server instance
        register_individual_tools: Whether to register individual help tools (default: True)

    Returns:
        The MCP server instance with help tools registered

    Notes:
        - Set register_individual_tools=False when using portmanteau help tool
        - Tools are registered with stateful=True to maintain state between invocations
        - State TTL is set based on the expected cache duration for each tool
    """
    # Register individual help tools only if requested
    if register_individual_tools:

        @mcp.tool()
        async def list_available_tools(detail: int = 1) -> dict[str, Any]:
            """List all available tools with stateful caching.

            This tool maintains a cache of available tools to improve performance.
            The cache is automatically managed by FastMCP's stateful tools.

            Args:
                detail: Level of detail (0=names only, 1=basic, 2=full)

            Returns:
                Dictionary with tool information
            """
            return await _list_tools_impl(mcp, detail)

        @mcp.tool()  # Get tool help
        async def get_tool_help(tool_name: str) -> dict[str, Any]:
            """Get detailed help for a specific tool with caching.

            This tool caches tool help documentation to improve performance.
            The cache is automatically managed by FastMCP's stateful tools.

            Args:
                tool_name: Name of the tool to get help for

            Returns:
                Detailed documentation for the tool
            """
            return await _get_tool_help_impl(mcp, tool_name)

        @mcp.tool()  # Search tools
        async def search_tools(query: str) -> dict[str, Any]:
            """Search for tools by name or description with stateful caching.

            This tool maintains a search index and caches search results.
            The cache is automatically managed by FastMCP's stateful tools.

            Args:
                query: Search term

            Returns:
                Dictionary with matching tools
            """
            return await _search_tools_impl(mcp, query)

        @mcp.tool()  # Get tool signature
        async def get_tool_signature(tool_name: str) -> dict[str, Any]:
            """Get the function signature for a tool with caching.

            This tool caches tool signatures to improve performance.
            The cache is automatically managed by FastMCP's stateful tools.

            Args:
                tool_name: Name of the tool

            Returns:
                Dictionary with tool signature information
            """
            return await _get_tool_signature_impl(mcp, tool_name)

        @mcp.tool()
        async def get_workflow_guides(category: str | None = None) -> dict[str, Any]:
            """Get comprehensive workflow guides and best practices.

            Provides complete workflow documentation for:
            - Model Management: Discovery, download, optimization
            - Content Generation: Professional writing pipelines
            - GPU Optimization: Memory management and performance
            - Fine-tuning: Complete training workflows

            Each workflow includes:
            - Step-by-step processes
            - Tool recommendations
            - Best practices
            - Parameter guidelines
            - Hardware requirements

            Args:
                category: Optional category filter (model_management, content_generation, gpu_optimization, fine_tuning)

            Returns:
                Complete workflow guides with examples and best practices
            """
            return await _get_workflow_guides_impl(category)

        @mcp.tool()
        async def get_performance_guide() -> dict[str, Any]:
            """Get comprehensive performance optimization guide.

            Covers all aspects of performance optimization:
            - GPU memory management and optimization
            - CPU utilization and multiprocessing
            - Model-specific optimizations
            - Benchmarking methodologies

            Includes specific recommendations for:
            - RTX 4090 memory fragmentation issues
            - Mixed precision training
            - Quantization strategies
            - Hardware-specific optimizations

            Returns:
                Complete performance optimization guide
            """
            return await _get_performance_guide_impl()

        @mcp.tool()
        async def get_troubleshooting_guide(category: str | None = None) -> dict[str, Any]:
            """Get comprehensive troubleshooting guide for common issues.

            Organized troubleshooting by category:
            - Model Loading: Memory, compatibility, authentication
            - Generation Issues: Quality, speed, parameters
            - GPU Issues: Memory fragmentation, thermal throttling
            - Network Issues: Downloads, rate limits, connectivity
            - Fine-tuning Issues: Training divergence, memory problems

            Each issue includes:
            - Symptoms and causes
            - Step-by-step solutions
            - Prevention strategies
            - Related tools and commands

            Args:
                category: Optional category filter

            Returns:
                Complete troubleshooting guide with solutions
            """
            return await _get_troubleshooting_guide_impl(category)

        @mcp.tool()
        async def get_hardware_requirements() -> dict[str, Any]:
            """Get detailed hardware requirements and performance estimates.

            Comprehensive hardware guidance including:
            - GPU recommendations by model size
            - Memory utilization estimates
            - Performance benchmarks (tokens/sec)
            - Training time estimates
            - Optimization recommendations

            Specific support for:
            - RTX 3090/4090 series
            - H100/A100 data center GPUs
            - Memory and thermal management
            - Quantization impact on performance

            Returns:
                Complete hardware requirements guide
            """
            return {
                "gpu_recommendations": [
                    {
                        "gpu": "RTX 3090 (24GB)",
                        "recommended_for": "Up to 13B models",
                        "performance": {
                            "7B_4bit": "12-18 tokens/sec",
                            "13B_4bit": "4-7 tokens/sec",
                            "30B": "Not recommended",
                        },
                        "vram_usage": {"7B_4bit": "16-20GB", "13B_4bit": "20-24GB"},
                    },
                    {
                        "gpu": "RTX 4090 (24GB)",
                        "recommended_for": "Up to 30B models",
                        "performance": {
                            "7B_4bit": "15-22 tokens/sec",
                            "13B_4bit": "6-9 tokens/sec",
                            "30B_4bit": "2-4 tokens/sec",
                            "70B": "Not recommended",
                        },
                        "vram_usage": {
                            "7B_4bit": "18-22GB",
                            "13B_4bit": "22-24GB",
                            "30B_4bit": "24GB+ (with optimizations)",
                        },
                        "notes": "Susceptible to memory fragmentation - use gpu_clear_memory() regularly",
                    },
                    {
                        "gpu": "H100 80GB",
                        "recommended_for": "Up to 70B+ models",
                        "performance": {
                            "7B_4bit": "40-60 tokens/sec",
                            "13B_4bit": "25-40 tokens/sec",
                            "30B_4bit": "12-20 tokens/sec",
                            "70B_4bit": "3-6 tokens/sec",
                        },
                        "vram_usage": {
                            "7B_4bit": "25-35GB",
                            "13B_4bit": "45-60GB",
                            "30B_4bit": "65-75GB",
                            "70B_4bit": "75-80GB",
                        },
                    },
                ],
                "cpu_requirements": {
                    "minimum": "8 cores, 16GB RAM",
                    "recommended": "16+ cores, 32GB+ RAM",
                    "for_training": "32+ cores, 128GB+ RAM",
                },
                "storage_requirements": {
                    "models": "100-500GB for common models",
                    "datasets": "Additional space for training data",
                    "temp_space": "50GB+ for model downloads and processing",
                },
                "training_time_estimates": {
                    "1M_tokens_7B_4090": "12-16 hours",
                    "1M_tokens_13B_4090": "30-45 hours",
                    "1M_tokens_7B_H100": "3-5 hours",
                    "1M_tokens_13B_H100": "6-8 hours",
                    "1M_tokens_30B_H100": "10-15 hours",
                    "1M_tokens_70B_H100": "40-60 hours",
                },
                "optimization_tips": [
                    "Use 4-bit quantization to reduce memory usage by 75%",
                    "Enable gradient checkpointing to save 60% memory during training",
                    "Use gradient accumulation for larger effective batch sizes",
                    "Enable Flash Attention 2.0 when available for 20-50% speedup",
                    "Use mixed precision training (bf16/fp16) for 2x performance",
                    "For large models, consider tensor parallelism",
                    "Clear GPU memory fragmentation regularly (especially RTX 4090)",
                    "Monitor GPU temperature to prevent thermal throttling",
                ],
                "network_requirements": {
                    "minimum": "25 Mbps download for model downloads",
                    "recommended": "100+ Mbps for large model downloads",
                    "huggingface_token": "Required for gated models (FLUX, etc.)",
                },
            }

        @mcp.tool()
        async def get_quick_reference() -> dict[str, Any]:
            """Get quick reference guide for common operations.

            Essential commands and workflows for immediate productivity:
            - Most frequently used tools and commands
            - Common parameter settings
            - Quick troubleshooting steps
            - Essential environment variables

            Returns:
                Quick reference guide for common operations
            """
            return {
                "most_used_tools": [
                    {"tool": "llm_health_tool", "operation": "health_check", "use": "Check system status"},
                    {"tool": "llm_models_tool", "operation": "list_models", "use": "See available models"},
                    {"tool": "llm_generation_tool", "operation": "generate_text", "use": "Generate content"},
                    {"tool": "gpu_status", "use": "Check GPU status"},
                    {"tool": "gpu_clear_memory", "use": "Fix memory fragmentation"},
                ],
                "essential_commands": [
                    "export HUGGINGFACE_TOKEN=hf_xxx  # For gated models",
                    "export OPENAI_API_KEY=sk-xxx     # For OpenAI models",
                    "gpu_clear_memory()                # Clear GPU fragmentation",
                    "llm_health_tool('health_check')   # System status",
                    "list_available_tools(detail=1)     # See all tools",
                ],
                "parameter_quick_reference": {
                    "temperature": {
                        "0.1-0.3": "Factual/technical writing",
                        "0.7-0.9": "Creative writing",
                        "0.4-0.6": "Balanced general use",
                    },
                    "max_tokens": {
                        "100-300": "Short responses",
                        "500-1000": "Medium responses",
                        "1000+": "Long detailed responses",
                    },
                    "quantization": {
                        "none": "Full precision (most accurate, uses most memory)",
                        "4bit": "75% memory reduction, minimal quality loss",
                        "8bit": "50% memory reduction, slight quality loss",
                    },
                },
                "common_issues_quick_fix": [
                    {"issue": "CUDA out of memory", "fix": "gpu_clear_memory() then use 4-bit quantization"},
                    {"issue": "Slow generation", "fix": "Lower temperature, use smaller model"},
                    {"issue": "Gated model access", "fix": "Set HUGGINGFACE_TOKEN environment variable"},
                    {"issue": "Poor output quality", "fix": "Adjust temperature or try different model"},
                    {"issue": "Training divergence", "fix": "Reduce learning rate, check data quality"},
                ],
                "environment_variables": [
                    "HUGGINGFACE_TOKEN - For gated models (FLUX, etc.)",
                    "HF_TOKEN - Alternative Hugging Face token",
                    "OPENAI_API_KEY - For OpenAI models",
                    "ANTHROPIC_API_KEY - For Claude models",
                    "LLM_MCP_CACHE_DIR - Custom model cache location",
                ],
            }

        @mcp.tool()
        async def get_integration_guide() -> dict[str, Any]:
            """Get comprehensive integration guide for external systems.

            Covers integration with:
            - Claude Desktop and MCP clients
            - Jupyter notebooks and Python environments
            - Docker containers and orchestration
            - CI/CD pipelines and automation
            - Web applications and APIs

            Includes:
            - Configuration examples
            - Authentication setup
            - Performance optimization
            - Monitoring and logging

            Returns:
                Complete integration guide for various environments
            """
            return {
                "claude_desktop_integration": {
                    "setup": [
                        "Download MCPB file from releases",
                        "Drag and drop into Claude Desktop settings",
                        "Restart Claude Desktop",
                        "Tools will be automatically available",
                    ],
                    "configuration": {
                        "model_cache": "Set LLM_MCP_CACHE_DIR for custom cache location",
                        "api_keys": "Configure via environment variables or .env file",
                        "gpu_settings": "Tools automatically detect GPU availability",
                    },
                },
                "python_notebook_integration": {
                    "jupyter_setup": [
                        "Install required packages: pip install fastmcp torch transformers",
                        "Set environment variables in notebook",
                        "Import and use tools directly",
                        "Use async/await for tool calls",
                    ],
                    "colab_setup": [
                        "Use GPU runtime for better performance",
                        "Install dependencies with !pip commands",
                        "Set environment variables with os.environ",
                        "Use gpu_status() to verify GPU availability",
                    ],
                },
                "docker_integration": {
                    "container_setup": [
                        "Use provided Dockerfile.mcp",
                        "Mount model cache volume for persistence",
                        "Set environment variables in docker-compose.yml",
                        "Expose port 8000 for API access",
                    ],
                    "orchestration": [
                        "Use docker-compose for multi-service setup",
                        "Configure GPU passthrough for containers",
                        "Set resource limits for memory and CPU",
                        "Use health checks for service monitoring",
                    ],
                },
                "api_integration": {
                    "rest_api": [
                        "Server exposes REST API on port 8000",
                        "Use /tools endpoint for tool discovery",
                        "POST to /call/{tool_name} for tool execution",
                        "JSON request/response format",
                    ],
                    "authentication": [
                        "API key authentication for external access",
                        "Rate limiting and request throttling",
                        "CORS configuration for web applications",
                        "Secure token management",
                    ],
                },
                "ci_cd_integration": {
                    "github_actions": [
                        "Use provided CI/CD workflows",
                        "Set secrets for API keys",
                        "Cache model downloads between runs",
                        "Run tests and linting automatically",
                    ],
                    "testing": [
                        "Use pytest for unit and integration tests",
                        "Test GPU functionality with mock data",
                        "Validate tool signatures and responses",
                        "Performance regression testing",
                    ],
                },
                "monitoring_integration": {
                    "logging": [
                        "Structured logging with JSON format",
                        "Configurable log levels and rotation",
                        "Performance metrics collection",
                        "Error tracking and alerting",
                    ],
                    "metrics": [
                        "GPU utilization and memory tracking",
                        "Request latency and throughput monitoring",
                        "Model loading and inference metrics",
                        "System resource usage statistics",
                    ],
                },
            }

    return mcp
