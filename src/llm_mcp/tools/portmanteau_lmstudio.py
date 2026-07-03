"""LM Studio Portmanteau tool for Local LLM MCP server.

This tool consolidates all LM Studio operations into a single interface
following the portmanteau pattern.

PORTMANTEAU PATTERN RATIONALE:
Instead of creating 3 separate LM Studio tools (one per operation), this tool consolidates
related operations into a single interface. Prevents tool explosion (3 tools → 1 tool) while maintaining
full functionality and improving discoverability. Follows FastMCP 2.13+ best practices.
"""

from typing import Any

from llm_mcp.tools.model_management_tools import (
    _lmstudio_list_models_impl,
    _lmstudio_load_model_impl,
    _lmstudio_unload_model_impl,
)
from llm_mcp.utils.logging import get_logger

logger = get_logger(__name__)

# Import FastMCP components
try:
    from fastmcp import FastMCP
    from fastmcp.tools import Tool

    FASTMCP_AVAILABLE = True
except ImportError:
    logger.error("FastMCP not available - portmanteau tools require FastMCP >= 2.12.0")
    FASTMCP_AVAILABLE = False


async def llm_lmstudio(
    operation: str,
    # Model operations
    model_path: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Comprehensive LM Studio management tool for Local LLM MCP server.

    PORTMANTEAU PATTERN: Consolidates 3 LM Studio operations into one tool.

    SUPPORTED OPERATIONS:
    - list_models: List all loaded models in LM Studio
    - load_model: Load a model by path (requires model_path)
    - unload_model: Unload a model (requires model_name)

    Args:
        operation: Operation to perform (see SUPPORTED OPERATIONS above)
        model_path: File path to model for load operations
        model_name: Model identifier for unload operations

    Returns:
        Operation-specific result dictionary
    """
    try:
        if operation == "list_models":
            return await _lmstudio_list_models_impl()

        elif operation == "load_model":
            if not model_path:
                return {"error": "model_path required for load_model operation"}
            return await _lmstudio_load_model_impl(model_path)

        elif operation == "unload_model":
            if not model_name:
                return {"error": "model_name required for unload_model operation"}
            return await _lmstudio_unload_model_impl(model_name)

        else:
            return {
                "error": f"Unknown operation: {operation}",
                "available_operations": ["list_models", "load_model", "unload_model"],
            }

    except Exception as e:
        logger.error(f"Error in llm_lmstudio operation {operation}: {e}", exc_info=True)
        return {"error": f"Operation failed: {e!s}", "operation": operation}


def register_llm_lmstudio_tools(mcp):
    """Register the LM Studio Portmanteau tool with the MCP server."""
    if not FASTMCP_AVAILABLE:
        logger.error("Cannot register LM Studio tools - FastMCP not available")
        return mcp

    @mcp.tool()
    async def llm_lmstudio_tool(
        operation: str,
        model_path: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        """LM Studio Portmanteau Tool - Consolidated LM Studio operations.

        This tool consolidates all LM Studio operations into a single interface,
        reducing the number of MCP tools while maintaining full functionality.

        Use the 'operation' parameter to specify what you want to do:
        - list_models: List all loaded models
        - load_model: Load a model from file path (requires model_path parameter)
        - unload_model: Unload a model (requires model_name parameter)
        """
        return await llm_lmstudio(
            operation=operation,
            model_path=model_path,
            model_name=model_name,
        )

    logger.info("Registered LM Studio portmanteau tool")
    return mcp
