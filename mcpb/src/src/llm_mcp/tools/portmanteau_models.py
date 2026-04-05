"""LLM Models portmanteau tool for Local LLM MCP server.

This tool consolidates all model management, registration, and basic provider operations
into a single interface following the portmanteau pattern.
"""

import importlib.util
from typing import Dict, Any, Optional

from llm_mcp.tools.model_tools import (
    _list_models_impl,
    _get_model_info_impl,
    _register_model_impl,
)
from llm_mcp.tools.model_management_tools import (
    _ollama_list_models_impl,
    _ollama_pull_model_impl,
    _ollama_delete_model_impl,
    _ollama_load_model_impl,
    _ollama_unload_model_impl,
    _lmstudio_list_models_impl,
    _lmstudio_load_model_impl,
    _lmstudio_unload_model_impl,
)
from llm_mcp.tools.vllm_tools import (
    _vllm_list_models_impl,
    _vllm_initialize_impl,
    _vllm_unload_impl,
)
from llm_mcp.utils.logging import get_logger

logger = get_logger(__name__)


# Check for FastMCP availability
FASTMCP_AVAILABLE = importlib.util.find_spec("fastmcp") is not None

if not FASTMCP_AVAILABLE:
    logger.error("FastMCP not available - portmanteau tools require FastMCP >= 2.12.0")


async def llm_models(
    operation: str,
    # Model operations
    model_id: Optional[str] = None,
    provider: Optional[str] = None,
    # Registration operations
    name: Optional[str] = None,
    context_length: Optional[int] = None,
    max_tokens: Optional[int] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Comprehensive model management tool for Local LLM MCP server.

    PORTMANTEAU PATTERN: Consolidates 8+ model operations into one tool.

    SUPPORTED OPERATIONS:
    - list_models: List all available models (optional provider filter)
    - get_model_info: Get detailed information for specific model (requires model_id)
    - register_model: Register a new model (requires name, context_length, max_tokens)
    - ollama_list: List available Ollama models
    - ollama_pull: Download/pull Ollama model (requires model_id as model_name)
    - ollama_delete: Delete Ollama model (requires model_id as model_name)
    - ollama_load: Load Ollama model for inference (requires model_id as model_name)
    - ollama_unload: Unload currently loaded Ollama model
    - lmstudio_list: List available LM Studio models
    - lmstudio_load: Load LM Studio model (requires model_id as model_name)
    - lmstudio_unload: Unload currently loaded LM Studio model
    - vllm_list: List currently loaded vLLM model
    - vllm_load: Initialize/load vLLM engine (requires model_id as model_name)
    - vllm_unload: Unload vLLM engine from memory

    Args:
        operation: Operation to perform (see SUPPORTED OPERATIONS above)
        model_id: Model identifier (required for get_model_info, register operations)
        provider: Provider filter for list_models
        name: Human-readable name for register_model
        context_length: Maximum context length for register_model
        max_tokens: Maximum tokens for register_model
        description: Optional description for register_model
        parameters: Optional parameters dict for register_model

    Returns:
        Operation-specific result dictionary
    """
    try:
        if operation == "list_models":
            return await _list_models_impl(provider)

        elif operation == "get_model_info":
            if not model_id:
                return {"error": "model_id required for get_model_info operation"}
            return await _get_model_info_impl(model_id)

        elif operation == "register_model":
            if not all([name, context_length is not None, max_tokens is not None]):
                return {
                    "error": "name, context_length, and max_tokens required for register_model operation"
                }
            return await _register_model_impl(
                model_id=model_id or f"custom_{name.lower().replace(' ', '_')}",
                name=name,
                provider=provider or "custom",
                context_length=context_length,
                max_tokens=max_tokens,
                description=description,
                parameters=parameters or {},
            )

        elif operation == "ollama_list":
            return await _ollama_list_models_impl()

        elif operation == "ollama_pull":
            if not model_id:
                return {
                    "error": "model_id required for ollama_pull operation (use as model_name)"
                }
            return await _ollama_pull_model_impl(model_id)

        elif operation == "ollama_delete":
            if not model_id:
                return {
                    "error": "model_id required for ollama_delete operation (use as model_name)"
                }
            return await _ollama_delete_model_impl(model_id)

        elif operation == "ollama_load":
            if not model_id:
                return {
                    "error": "model_id required for ollama_load operation (use as model_name)"
                }
            return await _ollama_load_model_impl(model_id)

        elif operation == "ollama_unload":
            return await _ollama_unload_model_impl()

        elif operation == "lmstudio_list":
            return await _lmstudio_list_models_impl()

        elif operation == "lmstudio_load":
            if not model_id:
                return {
                    "error": "model_id required for lmstudio_load operation (use as model_name)"
                }
            return await _lmstudio_load_model_impl(model_id)

        elif operation == "lmstudio_unload":
            return await _lmstudio_unload_model_impl()

        elif operation == "vllm_list":
            return await _vllm_list_models_impl()

        elif operation == "vllm_load":
            if not model_id:
                return {"error": "model_id required for vllm_load operation"}
            # Map additional parameters for vLLM initialization
            vllm_kwargs = {}
            if parameters:
                vllm_kwargs.update(parameters)
            return await _vllm_initialize_impl(model_id, **vllm_kwargs)

        elif operation == "vllm_unload":
            return await _vllm_unload_impl()

        else:
            return {
                "error": f"Unknown operation: {operation}",
                "available_operations": [
                    "list_models",
                    "get_model_info",
                    "register_model",
                    "ollama_list",
                    "ollama_pull",
                    "ollama_delete",
                    "ollama_load",
                    "ollama_unload",
                    "lmstudio_list",
                    "lmstudio_load",
                    "lmstudio_unload",
                    "vllm_list",
                    "vllm_load",
                    "vllm_unload",
                ],
            }

    except Exception as e:
        logger.error(f"Error in llm_models operation {operation}: {e}", exc_info=True)
        return {"error": f"Operation failed: {str(e)}", "operation": operation}


def register_llm_models_tools(mcp):
    """Register the LLM Models portmanteau tool with the MCP server."""
    if not FASTMCP_AVAILABLE:
        logger.error("Cannot register LLM Models tools - FastMCP not available")
        return mcp

    @mcp.tool()
    async def llm_models_tool(
        operation: str,
        model_id: Optional[str] = None,
        provider: Optional[str] = None,
        name: Optional[str] = None,
        context_length: Optional[int] = None,
        max_tokens: Optional[int] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """LLM Models Portmanteau Tool - Consolidated model management operations.

        This tool consolidates all model management, registration, and basic provider operations
        into a single interface, reducing the number of MCP tools while maintaining full functionality.

        Use the 'operation' parameter to specify what you want to do:
        - list_models: List all available models (filter by provider)
        - get_model_info: Get detailed info for specific model
        - register_model: Register a new custom model
        - ollama_*: Ollama-specific operations (list, pull, delete, load, unload)
        - lmstudio_*: LM Studio-specific operations (list, load, unload)
        - vllm_*: vLLM-specific operations (list, load, unload)
        """
        return await llm_models(
            operation=operation,
            model_id=model_id,
            provider=provider,
            name=name,
            context_length=context_length,
            max_tokens=max_tokens,
            description=description,
            parameters=parameters,
        )

    logger.info("Registered LLM Models portmanteau tool")
    return mcp
