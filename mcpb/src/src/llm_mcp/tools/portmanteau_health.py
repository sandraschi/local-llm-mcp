"""LLM Health portmanteau tool for Local LLM MCP server.

This tool consolidates all health, monitoring, system, and server management operations
into a single interface following the portmanteau pattern from Advanced Memory MCP.
"""

from typing import Any

from llm_mcp.tools.help_tools import _get_tool_help_impl, _get_tool_signature_impl, _list_tools_impl, _search_tools_impl
from llm_mcp.tools.monitoring_tools import (
    collect_system_metrics,
    get_metric_stats_impl,
    get_metrics_impl,
    set_log_level_impl,
)
from llm_mcp.tools.system_tools import get_service_status, get_system_info
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

# Global MCP instance (set during registration)
_mcp = None


async def llm_health(
    operation: str,
    # Help operations
    detail: int = 1,
    tool_name: str | None = None,
    query: str | None = None,
    # Metrics operations
    name: str | None = None,
    since_minutes: float = 60.0,
    tags: dict[str, str] | None = None,
    # Log operations
    logger_name: str = "",
    level: str = "INFO",
) -> dict[str, Any]:
    """Comprehensive health and system management tool for Local LLM MCP server.

    PORTMANTEAU PATTERN: Consolidates 15+ health/monitoring operations into one tool.

    SUPPORTED OPERATIONS:
    - health_check: Overall server health status
    - list_tools: List all available tools (detail: 0=names, 1=basic, 2=full)
    - tool_help: Get detailed help for specific tool (requires tool_name)
    - search_tools: Search tools by name/description (requires query)
    - tool_signature: Get function signature for tool (requires tool_name)
    - hardware_requirements: Get hardware requirements for fine-tuning
    - system_info: Get detailed system information
    - service_status: Check status of dependent services
    - server_health: Get comprehensive server health
    - get_metrics: Get metrics data (requires name, optional tags/since_minutes)
    - get_metric_stats: Get metric statistics (requires name, optional tags/since_minutes)
    - set_log_level: Set logging level (logger_name, level)
    - collect_metrics: Collect current system metrics

    Args:
        operation: Operation to perform (see SUPPORTED OPERATIONS above)
        detail: Detail level for list_tools (0=names only, 1=basic, 2=full)
        tool_name: Tool name for tool_help/tool_signature operations
        query: Search query for search_tools operation
        name: Metric name for get_metrics/get_metric_stats operations
        since_minutes: How far back to look for metrics (default: 60.0)
        tags: Optional tags to filter metrics
        logger_name: Logger name for set_log_level (empty string for root)
        level: Log level for set_log_level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Operation-specific result dictionary
    """
    global _mcp

    try:
        if operation == "health_check":
            # Overall server health check
            system_info = get_system_info()
            service_status = get_service_status()

            health_score = 100
            issues = []

            # Check CPU usage
            cpu_percent = system_info.get("cpu", {}).get("cpu_percent", [0])[0]
            if cpu_percent > 90:
                health_score -= 30
                issues.append(".1f")
            elif cpu_percent > 70:
                health_score -= 10
                issues.append(".1f")

            # Check memory usage
            memory_percent = system_info.get("memory", {}).get("percent", 0)
            if memory_percent > 95:
                health_score -= 30
                issues.append(".1f")
            elif memory_percent > 85:
                health_score -= 10
                issues.append(".1f")

            # Check disk usage
            disk_percent = system_info.get("disk", {}).get("percent", 0)
            if disk_percent > 95:
                health_score -= 20
                issues.append(".1f")

            # Check services
            for service_name, status_info in service_status.items():
                if isinstance(status_info, dict) and status_info.get("status") == "error":
                    health_score -= 15
                    issues.append(f"Service {service_name} is down")

            return {
                "status": "healthy" if health_score >= 80 else "degraded" if health_score >= 50 else "critical",
                "health_score": health_score,
                "timestamp": system_info.get("timestamp"),
                "issues": issues,
                "system": system_info,
                "services": service_status
            }

        elif operation == "list_tools":
            if not _mcp:
                return {"error": "MCP instance not available"}
            return await _list_tools_impl(_mcp, detail)

        elif operation == "tool_help":
            if not tool_name:
                return {"error": "tool_name required for tool_help operation"}
            if not _mcp:
                return {"error": "MCP instance not available"}
            return await _get_tool_help_impl(_mcp, tool_name)

        elif operation == "search_tools":
            if not query:
                return {"error": "query required for search_tools operation"}
            if not _mcp:
                return {"error": "MCP instance not available"}
            return await _search_tools_impl(_mcp, query)

        elif operation == "tool_signature":
            if not tool_name:
                return {"error": "tool_name required for tool_signature operation"}
            if not _mcp:
                return {"error": "MCP instance not available"}
            return await _get_tool_signature_impl(_mcp, tool_name)

        elif operation == "hardware_requirements":
            return {
                "fine_tuning_requirements": {
                    "min_gpu_memory": "8GB",
                    "recommended_gpu_memory": "24GB+",
                    "min_system_memory": "16GB",
                    "recommended_system_memory": "64GB+",
                    "supported_gpus": ["NVIDIA RTX 30xx+", "RTX 40xx+", "A-series", "H-series"],
                    "cpu_cores": "8+ recommended",
                    "storage": "100GB+ for models and datasets"
                },
                "inference_requirements": {
                    "min_gpu_memory": "4GB",
                    "recommended_gpu_memory": "8GB+",
                    "min_system_memory": "8GB",
                    "cpu_cores": "4+ recommended"
                }
            }

        elif operation == "system_info":
            return get_system_info()

        elif operation == "service_status":
            return get_service_status()

        elif operation == "server_health":
            return await llm_health("health_check")

        elif operation == "get_metrics":
            if not name:
                return {"error": "name required for get_metrics operation"}
            return await get_metrics_impl(name, since_minutes, tags)

        elif operation == "get_metric_stats":
            if not name:
                return {"error": "name required for get_metric_stats operation"}
            return await get_metric_stats_impl(name, since_minutes, tags)

        elif operation == "set_log_level":
            return await set_log_level_impl(logger_name, level)

        elif operation == "collect_metrics":
            return await collect_system_metrics()

        else:
            return {
                "error": f"Unknown operation: {operation}",
                "available_operations": [
                    "health_check", "list_tools", "tool_help", "search_tools", "tool_signature",
                    "hardware_requirements", "system_info", "service_status", "server_health",
                    "get_metrics", "get_metric_stats", "set_log_level", "collect_metrics"
                ]
            }

    except Exception as e:
        logger.error(f"Error in llm_health operation {operation}: {e}", exc_info=True)
        return {"error": f"Operation failed: {e!s}", "operation": operation}


def register_llm_health_tools(mcp):
    """Register the LLM Health portmanteau tool with the MCP server."""
    global _mcp
    _mcp = mcp

    if not FASTMCP_AVAILABLE:
        logger.error("Cannot register LLM Health tools - FastMCP not available")
        return mcp

    @mcp.tool()
    async def llm_health_tool(
        operation: str,
        detail: int = 1,
        tool_name: str | None = None,
        query: str | None = None,
        name: str | None = None,
        since_minutes: float = 60.0,
        tags: dict[str, str] | None = None,
        logger_name: str = "",
        level: str = "INFO",
    ) -> dict[str, Any]:
        """LLM Health Portmanteau Tool - Consolidated health, monitoring, and system operations.

        This tool consolidates all health, monitoring, system, and server management operations
        into a single interface, reducing the number of MCP tools while maintaining full functionality.

        Use the 'operation' parameter to specify what you want to do:
        - health_check: Overall server health status
        - list_tools: List all available tools
        - tool_help: Get help for specific tool
        - search_tools: Search for tools
        - tool_signature: Get tool function signature
        - hardware_requirements: Get hardware requirements
        - system_info: Get system information
        - service_status: Check service status
        - server_health: Comprehensive server health
        - get_metrics: Get metrics data
        - get_metric_stats: Get metric statistics
        - set_log_level: Set logging level
        - collect_metrics: Collect system metrics
        """
        return await llm_health(
            operation=operation,
            detail=detail,
            tool_name=tool_name,
            query=query,
            name=name,
            since_minutes=since_minutes,
            tags=tags,
            logger_name=logger_name,
            level=level,
        )

    logger.info("Registered LLM Health portmanteau tool")
    return mcp
