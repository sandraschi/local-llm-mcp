"""Help Portmanteau tool for Local LLM MCP server.

This tool consolidates all help and documentation operations into a single interface
following the portmanteau pattern.

PORTMANTEAU PATTERN RATIONALE:
Instead of creating 10+ separate help tools (one per operation), this tool consolidates
related help operations into a single interface. Prevents tool explosion (10 tools → 1 tool)
while maintaining full functionality and improving discoverability. Follows FastMCP 2.13+
best practices and provides comprehensive documentation system.
"""

from enum import Enum
from typing import Any

from llm_mcp.utils.logging import get_logger

logger = get_logger(__name__)


class HelpLevel(Enum):
    """Help detail levels for progressive disclosure."""

    NAMES_ONLY = 0  # Just tool names
    BASIC = 1  # Basic descriptions
    INTERMEDIATE = 2  # Examples and workflows
    ADVANCED = 3  # Performance & integration
    EXPERT = 4  # Architecture & troubleshooting


async def llm_help_tool(
    operation: str,
    # Tool discovery parameters
    detail: int = 1,
    # Tool help parameters
    tool_name: str | None = None,
    help_detail: int | None = None,
    # Search parameters
    query: str | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    """Comprehensive help and documentation system for Local LLM MCP server.

    PORTMANTEAU PATTERN: Consolidates 10+ help operations into one tool.

    SUPPORTED OPERATIONS:
    - list_tools: Tool discovery with 5 detail levels (0-4)
    - get_tool_help: Comprehensive documentation for any tool
    - search_tools: Relevance-scored tool search
    - get_tool_signature: Function signatures with metadata
    - get_workflow_guides: Complete workflow documentation
    - get_performance_guide: Performance optimization strategies
    - get_troubleshooting_guide: Comprehensive issue resolution
    - get_hardware_requirements: Hardware recommendations and limits
    - get_quick_reference: Essential commands and settings
    - get_integration_guide: External system integration guides

    Args:
        operation: Operation to perform (see SUPPORTED OPERATIONS above)
        detail: Detail level for list_tools (0-4)
        tool_name: Tool name for help operations
        help_detail: Detail level for tool help (0-4)
        query: Search query for search_tools
        category: Category filter for search_tools

    Returns:
        Operation-specific result dictionary
    """
    try:
        # Import here to avoid circular imports
        from llm_mcp.tools.help_tools import (
            _get_hardware_requirements,
            _get_integration_guide_impl,
            _get_performance_guide_impl,
            _get_quick_reference_impl,
            _get_tool_help_impl,
            _get_tool_signature_impl,
            _get_troubleshooting_guide_impl,
            _get_workflow_guides_impl,
            _list_tools_impl,
            _search_tools_impl,
        )

        if operation == "list_tools":
            return await _list_tools_impl(None, detail)

        elif operation == "get_tool_help":
            if not tool_name:
                return {"error": "tool_name required for get_tool_help operation"}
            detail_level = help_detail if help_detail is not None else 2
            return await _get_tool_help_impl(None, tool_name, detail_level)

        elif operation == "search_tools":
            if not query:
                return {"error": "query required for search_tools operation"}
            return await _search_tools_impl(None, query)

        elif operation == "get_tool_signature":
            if not tool_name:
                return {"error": "tool_name required for get_tool_signature operation"}
            return await _get_tool_signature_impl(None, tool_name)

        elif operation == "get_workflow_guides":
            return await _get_workflow_guides_impl(category)

        elif operation == "get_performance_guide":
            return await _get_performance_guide_impl()

        elif operation == "get_troubleshooting_guide":
            return await _get_troubleshooting_guide_impl(category)

        elif operation == "get_hardware_requirements":
            return _get_hardware_requirements()

        elif operation == "get_quick_reference":
            return await _get_quick_reference_impl()

        elif operation == "get_integration_guide":
            return await _get_integration_guide_impl()

        else:
            return {
                "error": f"Unknown operation: {operation}",
                "available_operations": [
                    "list_tools",
                    "get_tool_help",
                    "search_tools",
                    "get_tool_signature",
                    "get_workflow_guides",
                    "get_performance_guide",
                    "get_troubleshooting_guide",
                    "get_hardware_requirements",
                    "get_quick_reference",
                    "get_integration_guide",
                ],
            }

    except Exception as e:
        logger.error(f"Error in llm_help_tool operation {operation}: {e}", exc_info=True)
        return {"error": f"Operation failed: {e!s}", "operation": operation}


def register_llm_help_tools(mcp):
    """Register the Help Portmanteau tool with the MCP server."""

    @mcp.tool()
    async def llm_help_tool_portmanteau(
        operation: str,
        detail: int = 1,
        tool_name: str | None = None,
        help_detail: int | None = None,
        query: str | None = None,
        category: str | None = None,
    ) -> dict[str, Any]:
        """Help Portmanteau Tool - Consolidated help and documentation operations.

        This tool consolidates all help and documentation operations into a single interface,
        providing comprehensive assistance for using the Local LLM MCP Server.

        Use the 'operation' parameter to specify what you want to do:
        - list_tools: Tool discovery with 5 detail levels (0-4)
        - get_tool_help: Comprehensive documentation for any tool
        - search_tools: Relevance-scored tool search
        - get_tool_signature: Function signatures with metadata
        - get_workflow_guides: Complete workflow documentation
        - get_performance_guide: Performance optimization strategies
        - get_troubleshooting_guide: Comprehensive issue resolution
        - get_hardware_requirements: Hardware recommendations and limits
        - get_quick_reference: Essential commands and settings
        - get_integration_guide: External system integration guides
        """
        return await llm_help_tool(
            operation=operation,
            detail=detail,
            tool_name=tool_name,
            help_detail=help_detail,
            query=query,
            category=category,
        )

    logger.info("Registered Help portmanteau tool")
    return mcp
