"""vLLM Portmanteau tool for Local LLM MCP server.

This tool consolidates all vLLM operations into a single interface
following the portmanteau pattern.

PORTMANTEAU PATTERN RATIONALE:
Instead of creating 5+ separate vLLM tools (one per operation), this tool consolidates
related operations into a single interface. Prevents tool explosion (5+ tools → 1 tool) while maintaining
full functionality and improving discoverability. Follows FastMCP 2.13+ best practices.
"""

from typing import Any

# Try to import vLLM functions
try:
    from llm_mcp.tools.vllm_tools import (
        _vllm_get_status_impl,
        _vllm_list_models_impl,
        _vllm_load_model_impl,
        _vllm_unload_model_impl,
    )

    VLLM_FUNCTIONS_AVAILABLE = True
except ImportError:
    VLLM_FUNCTIONS_AVAILABLE = False

from llm_mcp.utils.logging import get_logger

logger = get_logger(__name__)

# Import FastMCP components
try:
    from fastmcp import FastMCP  # noqa: F401
    from fastmcp.tools import Tool  # noqa: F401

    FASTMCP_AVAILABLE = True
except ImportError:
    logger.error("FastMCP not available - portmanteau tools require FastMCP >= 2.12.0")
    FASTMCP_AVAILABLE = False


async def llm_vllm(
    operation: str,
    # Model operations
    model_name: str | None = None,
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Comprehensive vLLM management tool for Local LLM MCP server.

    PORTMANTEAU PATTERN: Consolidates 5+ vLLM operations into one tool.

    SUPPORTED OPERATIONS:
    - list_models: List all loaded models in vLLM
    - load_model: Load a model with configuration (requires model_name)
    - unload_model: Unload a model (requires model_name)
    - get_status: Get vLLM server status
    - get_metrics: Get performance metrics

    Args:
        operation: Operation to perform (see SUPPORTED OPERATIONS above)
        model_name: Model name for operations
        model_config: Configuration dictionary for model loading

    Returns:
        Operation-specific result dictionary
    """
    try:
        if not VLLM_FUNCTIONS_AVAILABLE:
            return {"error": "vLLM functions not available - vLLM may not be installed"}

        if operation == "list_models":
            return await _vllm_list_models_impl()

        elif operation == "load_model":
            if not model_name:
                return {"error": "model_name required for load_model operation"}
            return await _vllm_load_model_impl(model_name, model_config or {})

        elif operation == "unload_model":
            if not model_name:
                return {"error": "model_name required for unload_model operation"}
            return await _vllm_unload_model_impl(model_name)

        elif operation == "get_status":
            return await _vllm_get_status_impl()

        elif operation == "get_metrics":
            # This would need to be implemented in vllm_tools.py
            return {"error": "get_metrics operation not yet implemented"}

        else:
            return {
                "error": f"Unknown operation: {operation}",
                "available_operations": ["list_models", "load_model", "unload_model", "get_status", "get_metrics"],
            }

    except Exception as e:
        logger.error(f"Error in llm_vllm operation {operation}: {e}", exc_info=True)
        return {"error": f"Operation failed: {e!s}", "operation": operation}


def register_llm_vllm_tools(mcp):
    """Register the vLLM Portmanteau tool with the MCP server."""
    if not FASTMCP_AVAILABLE:
        logger.error("Cannot register vLLM tools - FastMCP not available")
        return mcp

    @mcp.tool()
    async def llm_vllm_tool(
        operation: str,
        model_name: str | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """vLLM Portmanteau Tool - Consolidated vLLM operations.

        This tool consolidates all vLLM operations into a single interface,
        reducing the number of MCP tools while maintaining full functionality.

        Use the 'operation' parameter to specify what you want to do:
        - list_models: List all loaded models
        - load_model: Load a model (requires model_name parameter)
        - unload_model: Unload a model (requires model_name parameter)
        - get_status: Get vLLM server status
        - get_metrics: Get performance metrics
        """
        return await llm_vllm(
            operation=operation,
            model_name=model_name,
            model_config=model_config,
        )

    logger.info("Registered vLLM portmanteau tool")
    return mcp
