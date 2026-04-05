"""GPU Portmanteau tool for Local LLM MCP server.

This tool consolidates all GPU management operations into a single interface
following the portmanteau pattern.

PORTMANTEAU PATTERN RATIONALE:
Instead of creating 4+ separate GPU tools (one per operation), this tool consolidates
related GPU operations into a single interface. Prevents tool explosion (4 tools → 1 tool)
while maintaining full functionality and improving discoverability. Follows FastMCP 2.13+
best practices and provides comprehensive GPU management for NVIDIA GPUs.
"""

import logging
from typing import Dict, Any, Optional

from llm_mcp.utils.logging import get_logger

logger = get_logger(__name__)

def llm_gpu_tool(
    operation: str,
    # GPU identification
    gpu_id: int = 0,
) -> Dict[str, Any]:
    """Comprehensive GPU management tool for Local LLM MCP server.

    PORTMANTEAU PATTERN: Consolidates 4+ GPU operations into one tool.

    SUPPORTED OPERATIONS:
    - get_status: Comprehensive GPU monitoring and statistics
    - clear_memory: Clear GPU memory to prevent fragmentation
    - optimize: Advanced GPU memory optimization and health check
    - get_health: Detailed GPU health monitoring and diagnostics

    Args:
        operation: Operation to perform (see SUPPORTED OPERATIONS above)
        gpu_id: GPU device ID (default: 0)

    Returns:
        Operation-specific result dictionary
    """
    try:
        # Import GPU management functions
        from llm_mcp.tools.gpu_manager import (
            get_gpu_status, clear_gpu_memory, optimize_gpu_memory, monitor_gpu_health
        )

        if operation == "get_status":
            return get_gpu_status(gpu_id)

        elif operation == "clear_memory":
            return clear_gpu_memory(gpu_id)

        elif operation == "optimize":
            return optimize_gpu_memory(gpu_id)

        elif operation == "get_health":
            return monitor_gpu_health(gpu_id)

        else:
            return {
                "error": f"Unknown operation: {operation}",
                "available_operations": [
                    "get_status", "clear_memory", "optimize", "get_health"
                ]
            }

    except Exception as e:
        logger.error(f"Error in llm_gpu_tool operation {operation}: {e}", exc_info=True)
        return {"error": f"Operation failed: {str(e)}", "operation": operation}


def register_llm_gpu_tools(mcp):
    """Register the GPU Portmanteau tool with the MCP server."""
    @mcp.tool()
    async def llm_gpu_tool_portmanteau(
        operation: str,
        gpu_id: int = 0,
    ) -> Dict[str, Any]:
        """GPU Portmanteau Tool - Consolidated GPU management operations.

        This tool consolidates all GPU management operations into a single interface,
        providing comprehensive NVIDIA GPU monitoring and optimization for RTX series GPUs.

        Use the 'operation' parameter to specify what you want to do:
        - get_status: Comprehensive GPU monitoring and statistics
        - clear_memory: Clear GPU memory to prevent fragmentation (RTX 4090 optimized)
        - optimize: Advanced GPU memory optimization and health check
        - get_health: Detailed GPU health monitoring and diagnostics
        """
        return llm_gpu_tool(
            operation=operation,
            gpu_id=gpu_id,
        )

    logger.info("Registered GPU portmanteau tool")
    return mcp