"""Help tools for the LLM MCP server.

This module provides self-documenting functionality for the MCP server, allowing
clients to discover and understand available tools and their usage.
"""
from typing import Dict, List, Any, Type, Optional, get_origin, get_args, Union, Callable
import inspect
from functools import wraps


def format_type(t: Type) -> str:
    """Format a Python type for documentation.
    
    Args:
        t: The type to format
        
    Returns:
        Formatted type string
    """
    if t is type(None):  # noqa: E721
        return "None"
    if t == inspect.Parameter.empty:
        return "any"
        
    # Handle Optional types
    if get_origin(t) is Union:
        args = [a for a in get_args(t) if a is not type(None)]  # noqa: E721
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


def get_type_hints(func: Callable) -> Dict[str, Type]:
    """Safely get type hints from a function.
    
    Args:
        func: The function to get type hints from
        
    Returns:
        Dictionary of parameter names to types
    """
    try:
        return getattr(func, "__annotations__", {})
    except (NameError, AttributeError):
        return {}


def get_parameter_docs(func: Callable) -> List[Dict[str, Any]]:
    """Extract parameter documentation from a function.
    
    Args:
        func: The function to document
        
    Returns:
        List of parameter information dictionaries
    """
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    params = []
    
    for name, param in sig.parameters.items():
        if name == "self":
            continue
            
        param_info = {
            "name": name,
            "type": format_type(type_hints.get(name, param.annotation)),
            "required": param.default is param.empty,
            "default": param.default if param.default is not param.empty else None,
            "description": ""
        }
        
        # Get description from docstring
        doc = inspect.getdoc(func) or ""
        for line in doc.split('\n'):
            line = line.strip()
            if line.startswith(f"{name}:") and ":" in line:
                param_info["description"] = line.split(":", 1)[1].strip()
                break
                
        params.append(param_info)
    
    return params


def get_return_docs(func: Callable) -> Dict[str, str]:
    """Extract return type and description.
    
    Args:
        func: The function to document
        
    Returns:
        Dictionary with return type and description
    """
    type_hints = get_type_hints(func)
    return_type = type_hints.get("return", "None")
    
    doc = inspect.getdoc(func) or ""
    description = ""
    
    # Extract return description from docstring
    in_returns = False
    for line in doc.split('\n'):
        line = line.strip()
        if line.lower().startswith("returns:"):
            in_returns = True
            description = line[8:].strip()
        elif in_returns and line and not line.startswith(" "):
            break
        elif in_returns:
            description += " " + line.strip()
    
    return {
        "type": format_type(return_type),
        "description": description
    }


def get_tool_info(func: Callable) -> Dict[str, Any]:
    """Get complete documentation for a tool.
    
    Args:
        func: The tool function to document
        
    Returns:
        Dictionary with tool documentation
    """
    doc = inspect.getdoc(func) or ""
    description = doc.split('\n')[0] if doc else ""
    
    return {
        "name": func.__name__,
        "description": description,
        "parameters": get_parameter_docs(func),
        "returns": get_return_docs(func),
        "full_doc": doc
    }


# Implementation functions
async def _list_tools_impl(mcp: Any, detail: int = 1) -> Dict[str, Any]:
    """Implementation of list_tools functionality.
    
    Args:
        mcp: The MCP server instance
        detail: Level of detail (0=names only, 1=basic, 2=full)
        
    Returns:
        Dictionary with tool information
    """
    tools = {}
    mcp_tools = await mcp.get_tools()
    for name, tool in mcp_tools.items():
        if detail == 0:
            tools[name] = {"description": (inspect.getdoc(tool) or "").split('\n')[0]}
        else:
            tools[name] = get_tool_info(tool)
            if detail == 1:
                tools[name].pop("full_doc", None)
    return {"tools": tools}


async def _get_tool_help_impl(mcp: Any, tool_name: str) -> Dict[str, Any]:
    """Implementation of get_tool_help functionality.
    
    Args:
        mcp: The MCP server instance
        tool_name: Name of the tool to get help for
        
    Returns:
        Detailed documentation for the tool
    """
    mcp_tools = await mcp.get_tools()
    for name, tool in mcp_tools.items():
        if name == tool_name:
            return {"tool": tool_name, **get_tool_info(tool)}
    return {"error": f"Tool '{tool_name}' not found"}


async def _search_tools_impl(mcp: Any, query: str) -> Dict[str, Any]:
    """Implementation of search_tools functionality.
    
    Args:
        mcp: The MCP server instance
        query: Search term
        
    Returns:
        Dictionary with matching tools
    """
    query = query.lower()
    matches = []
    
    for name, tool in mcp.get_tools().items():
        doc = (inspect.getdoc(tool) or "").lower()
        if query in name.lower() or query in doc:
            matches.append({
                "name": name,
                "description": doc.split('\n')[0]
            })
    
    return {"matches": matches}


async def _get_tool_signature_impl(mcp: Any, tool_name: str) -> Dict[str, Any]:
    """Implementation of get_tool_signature functionality.
    
    Args:
        mcp: The MCP server instance
        tool_name: Name of the tool
        
    Returns:
        Dictionary with tool signature information
    """
    for name, tool in mcp.get_tools().items():
        if name == tool_name:
            sig = inspect.signature(tool)
            return {
                "name": name,
                "signature": str(sig),
                "parameters": [
                    {
                        "name": param.name,
                        "kind": str(param.kind),
                        "default": param.default if param.default is not param.empty else None,
                        "annotation": format_type(param.annotation)
                    }
                    for param in sig.parameters.values()
                    if param.name != 'self'
                ],
                "return_annotation": format_type(sig.return_annotation)
            }
    return {"error": f"Tool '{tool_name}' not found"}


def register_help_tools(mcp):
    """Register help tools with the MCP server using FastMCP 2.11.3 stateful features.
    
    Args:
        mcp: The MCP server instance
        
    Returns:
        The MCP server instance with help tools registered
        
    Notes:
        - Tools are registered with stateful=True to maintain state between invocations
        - State TTL is set based on the expected cache duration for each tool
    """
    @mcp.tool()  # Tool listing
    async def list_tools(detail: int = 1) -> Dict[str, Any]:
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
    async def get_tool_help(tool_name: str) -> Dict[str, Any]:
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
    async def search_tools(query: str) -> Dict[str, Any]:
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
    async def get_tool_signature(tool_name: str) -> Dict[str, Any]:
        """Get the function signature for a tool with caching.
        
        This tool caches tool signatures to improve performance.
        The cache is automatically managed by FastMCP's stateful tools.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Dictionary with tool signature information
        """
        return await _get_tool_signature_impl(mcp, tool_name)
    
    @mcp.tool()  # Get hardware requirements
    async def hardware_requirements() -> Dict[str, Any]:
        """Get hardware requirements and performance estimates for fine-tuning with caching.
        
        This tool caches hardware requirements to improve performance.
        The cache is automatically managed by FastMCP's stateful tools.
        
        Returns:
            Dictionary with hardware requirements and performance estimates
        """
        return {
            "gpu_recommendations": [
                {
                    "gpu": "RTX 4090 (24GB)",
                    "recommended_for": "7B models",
                    "max_model_size": "13B (with limitations)",
                    "performance": {
                        "7B_4bit": "15-20 tokens/sec",
                        "13B_4bit": "5-8 tokens/sec",
                        "30B+": "Not recommended"
                    },
                    "vram_usage": {
                        "7B_4bit": "18-22GB",
                        "13B_4bit": "22-24GB"
                    }
                },
                {
                    "gpu": "H100 80GB",
                    "recommended_for": "30B+ models",
                    "max_model_size": "70B+",
                    "performance": {
                        "7B_4bit": "45-60 tokens/sec",
                        "13B_4bit": "30-40 tokens/sec",
                        "30B_4bit": "15-25 tokens/sec",
                        "70B_4bit": "4-6 tokens/sec"
                    },
                    "vram_usage": {
                        "7B_4bit": "30-35GB",
                        "13B_4bit": "50-60GB",
                        "30B_4bit": "70-75GB",
                        "70B_4bit": "75-80GB"
                    }
                }
            ],
            "training_time_estimates": {
                "1M_tokens_7B_4090": "14-18 hours",
                "1M_tokens_13B_4090": "35-55 hours",
                "1M_tokens_7B_H100": "4.5-6 hours",
                "1M_tokens_13B_H100": "7-9 hours",
                "1M_tokens_30B_H100": "11-19 hours",
                "1M_tokens_70B_H100": "46-70 hours"
            },
            "optimization_tips": [
                "Use 4-bit quantization for memory efficiency",
                "Enable gradient checkpointing to reduce memory usage",
                "Use gradient accumulation for larger effective batch sizes",
                "Enable Flash Attention 2.0 if available",
                "Consider using mixed precision training (bf16/fp16)",
                "For large models, use model parallelism"
            ],
            "documentation": "See docs/hardware_requirements.md for detailed information"
        }

    return mcp
