"""Utility functions for tool registration and management."""
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')

def register_tool(
    mcp: Any,
    func: Callable[..., T],
    name: Optional[str] = None,
    description: Optional[str] = None,
    stateful: bool = False,
    ttl: Optional[int] = None
) -> Callable[..., T]:
    """Register a tool with the MCP server with standardized configuration.
    
    Args:
        mcp: The MCP server instance
        func: The function to register as a tool
        name: Optional name for the tool (defaults to function name)
        description: Optional description for the tool
        stateful: Whether the tool should maintain state between calls
        ttl: Time-to-live for cached results in seconds (if stateful)
        
    Returns:
        The decorated function
    """
    tool_name = name or func.__name__
    
    # Use provided description or extract from docstring
    if description is None:
        if func.__doc__:
            # Get the first non-empty line of the docstring
            doc_lines = [line.strip() for line in func.__doc__.split('\n') if line.strip()]
            description = doc_lines[0] if doc_lines else ""
        else:
            description = f"Tool: {tool_name}"
    
    # Create the decorator with appropriate parameters
    decorator = mcp.tool(
        name=tool_name,
        description=description,
        stateful=stateful,
        ttl=ttl
    )
    
    # Apply the decorator to the function
    decorated_func = decorator(func)
    
    # Log the registration
    logger.debug(
        f"Registered tool: {tool_name} "
        f"(stateful={stateful}, ttl={ttl or 'default'})"
    )
    
    return decorated_func

def register_tool_module(mcp: Any, module_name: str, register_func_name: str) -> Any:
    """Safely import and register a tool module.
    
    Args:
        mcp: The MCP server instance
        module_name: Name of the module containing the registration function
        register_func_name: Name of the registration function in the module
        
    Returns:
        The updated MCP server instance or None if registration failed
    """
    try:
        module = __import__(f"llm_mcp.tools.{module_name}", fromlist=[register_func_name])
        register_func = getattr(module, register_func_name)
        
        logger.info(f"Registering tools from {module_name}.{register_func_name}")
        mcp = register_func(mcp)
        return mcp
        
    except ImportError as e:
        logger.warning(f"Failed to import tool module {module_name}: {str(e)}")
    except AttributeError as e:
        logger.warning(f"Registration function {register_func_name} not found in {module_name}: {str(e)}")
    except Exception as e:
        logger.error(f"Error registering tools from {module_name}.{register_func_name}: {str(e)}", exc_info=True)
    
    return mcp

def get_tool_metadata(func: Callable) -> Dict[str, Any]:
    """Extract metadata from a tool function.
    
    Args:
        func: The tool function to extract metadata from
        
    Returns:
        Dictionary containing tool metadata
    """
    signature = inspect.signature(func)
    
    # Extract parameter information
    parameters = {}
    for name, param in signature.parameters.items():
        param_info = {
            "name": name,
            "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
            "default": param.default if param.default != inspect.Parameter.empty else None,
            "required": param.default == inspect.Parameter.empty and param.kind in [
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD
            ]
        }
        parameters[name] = param_info
    
    # Extract return type
    return_type = (
        str(signature.return_annotation)
        if signature.return_annotation != inspect.Signature.empty
        else "Any"
    )
    
    # Extract docstring
    docstring = func.__doc__ or ""
    
    return {
        "name": func.__name__,
        "docstring": docstring,
        "parameters": parameters,
        "return_type": return_type,
        "module": func.__module__
    }
