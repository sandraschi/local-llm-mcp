"""Ollama Portmanteau tool for Local LLM MCP server.

This tool consolidates all Ollama operations into a single interface
following the portmanteau pattern.

PORTMANTEAU PATTERN RATIONALE:
Instead of creating 5 separate Ollama tools (one per operation), this tool consolidates
related operations into a single interface. Prevents tool explosion (5 tools → 1 tool) while
maintaining full functionality and improving discoverability. Follows FastMCP 2.13+ best
practices.
"""

from typing import Any

from llm_mcp.tools.model_management_tools import (
    _ollama_delete_model_impl,
    _ollama_list_models_impl,
    _ollama_load_model_impl,
    _ollama_pull_model_impl,
    _ollama_unload_model_impl,
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


async def llm_ollama(
    operation: str,
    # Model operations
    model: str | None = None,
    model_path: str | None = None,
) -> dict[str, Any]:
    """Comprehensive Ollama management tool for Local LLM MCP server.

    PORTMANTEAU PATTERN: Consolidates 5 Ollama operations into one tool.

    SUPPORTED OPERATIONS:
    - list_models: List all available models in Ollama
    - pull_model: Download a model from Ollama registry (requires model)
    - load_model: Load a model into memory (requires model)
    - unload_model: Unload a model from memory (requires model)
    - delete_model: Delete a model from storage (requires model)

    Args:
        operation: Operation to perform (see SUPPORTED OPERATIONS above)
        model: Model name for operations (required for most operations)
        model_path: Optional path for model operations

    Returns:
        Operation-specific result dictionary
    """
    try:
        if operation == "list_models":
            return await _ollama_list_models_impl()

        elif operation == "pull_model":
            if not model:
                return {"error": "model required for pull_model operation"}
            return await _ollama_pull_model_impl(model)

        elif operation == "load_model":
            if not model:
                return {"error": "model required for load_model operation"}
            return await _ollama_load_model_impl(model)

        elif operation == "unload_model":
            if not model:
                return {"error": "model required for unload_model operation"}
            return await _ollama_unload_model_impl(model)

        elif operation == "delete_model":
            if not model:
                return {"error": "model required for delete_model operation"}
            return await _ollama_delete_model_impl(model)

        else:
            return {
                "error": f"Unknown operation: {operation}",
                "available_operations": ["list_models", "pull_model", "load_model", "unload_model", "delete_model"],
            }

    except Exception as e:
        logger.error(f"Error in llm_ollama operation {operation}: {e}", exc_info=True)
        return {"error": f"Operation failed: {e!s}", "operation": operation}


def register_llm_ollama_tools(mcp):
    """Register the Ollama Portmanteau tool with the MCP server."""
    if not FASTMCP_AVAILABLE:
        logger.error("Cannot register Ollama tools - FastMCP not available")
        return mcp

    @mcp.tool()
    async def llm_ollama_tool(
        operation: str,
        model: str | None = None,
        model_path: str | None = None,
    ) -> dict[str, Any]:
        """Ollama Portmanteau Tool - Consolidated Ollama operations.

        This tool consolidates all Ollama operations into a single interface,
        reducing the number of MCP tools while maintaining full functionality.

        Use the 'operation' parameter to specify what you want to do:
        - list_models: List all available models
        - pull_model: Download a model (requires model parameter)
        - load_model: Load a model into memory (requires model parameter)
        - unload_model: Unload a model from memory (requires model parameter)
        - delete_model: Delete a model from storage (requires model parameter)
        """
        return await llm_ollama(
            operation=operation,
            model=model,
            model_path=model_path,
        )

    logger.info("Registered Ollama portmanteau tool")
    return mcp
