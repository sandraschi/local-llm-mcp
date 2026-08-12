"""LM Studio Portmanteau tool for Local LLM MCP server.

This tool consolidates all LM Studio operations into a single interface
following the portmanteau pattern, including LM Link peer discovery.

PORTMANTEAU PATTERN RATIONALE:
Instead of creating separate LM Studio tools (one per operation), this tool consolidates
related operations into a single interface. Prevents tool explosion while maintaining
full functionality and improving discoverability. Follows FastMCP 2.13+ best practices.
"""

import asyncio
import json
import shutil
from typing import Any

from llm_mcp.tools.model_management_tools import (
    _lmstudio_list_models_impl,
    _lmstudio_load_model_impl,
    _lmstudio_unload_model_impl,
)
from llm_mcp.utils.logging import get_logger

logger = get_logger(__name__)


def _find_lms_binary() -> str | None:
    """Find the ``lms`` CLI binary on the system."""
    lms_in_path = shutil.which("lms")
    if lms_in_path:
        return lms_in_path
    import os

    candidates = [
        r"C:\Users\sandr\AppData\Local\Programs\lm-studio\lms.exe",
        r"C:\Program Files\lm-studio\lms.exe",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


async def _run_lms(args: list[str], timeout: int = 30) -> dict[str, Any]:
    """Run ``lms <args>`` via async subprocess, return structured result."""
    binary = _find_lms_binary()
    if binary is None:
        return {
            "ok": False,
            "error": "lms CLI not found",
            "error_type": "lms_not_found",
        }
    cmd = [binary, *args]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
        stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
        result: dict[str, Any] = {
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
        if stdout and proc.returncode == 0:
            try:
                result["parsed"] = json.loads(stdout)
            except json.JSONDecodeError:
                pass
        return result
    except TimeoutError:
        return {"ok": False, "error": f"lms timed out after {timeout}s", "error_type": "timeout"}
    except FileNotFoundError:
        return {"ok": False, "error": "lms binary disappeared", "error_type": "binary_missing"}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "error_type": "subprocess_error"}


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
    model_path: str | None = None,
    model_name: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Comprehensive LM Studio management tool for Local LLM MCP server.

    PORTMANTEAU PATTERN: Consolidates LM Studio operations into one tool,
    including local model management and LM Link peer discovery.

    SUPPORTED OPERATIONS:
    - list_models: List all loaded models in LM Studio
    - load_model: Load a model by path (requires model_path)
    - unload_model: Unload a model (requires model_name)
    - link_status: Show LM Link status -- peers, their loaded models, and
      link state. Uses ``lms link status --json``. LM Link (Feb 2026) is
      a Tailscale-powered encrypted mesh for remote LLM access.

    Args:
        operation: Operation to perform (see SUPPORTED OPERATIONS above)
        model_path: File path to model for load operations
        model_name: Model identifier for unload operations
        device: Remote device name for set_preferred_device (link_status only)

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
            return await _lmstudio_unload_model_impl(model_name)  # ty: ignore[too-many-positional-arguments]

        elif operation == "link_status":
            result = await _run_lms(["link", "status", "--json"])
            if result["ok"] and "parsed" in result:
                data = result["parsed"]
                return {
                    "success": True,
                    "operation": "link_status",
                    "message": f"LM Link: {data.get('connection_state', 'unknown')}",
                    "data": data,
                    "device_name": data.get("device_name", "unknown"),
                    "connected": data.get("enabled", False),
                    "peers": data.get("peers", []),
                    "peer_count": len(data.get("peers", [])),
                }
            if not result["ok"]:
                return {
                    "success": False,
                    "operation": "link_status",
                    "error": result.get("error") or result.get("stderr") or "lms link status failed",
                    "error_type": result.get("error_type", "cli_error"),
                    "message": "lms CLI not available or not logged in. Install LM Studio and run `lms login`.",
                    "recovery_options": [
                        "Install LM Studio from https://lmstudio.ai/download",
                        "Run `lms login` to authenticate with your LM Studio account",
                        "Run `lms link enable` to enable LM Link on this device",
                    ],
                }
            return {
                "success": True,
                "operation": "link_status",
                "message": "LM Link status returned but no parseable JSON",
                "raw_stdout": result.get("stdout", ""),
            }

        else:
            return {
                "error": f"Unknown operation: {operation}",
                "available_operations": ["list_models", "load_model", "unload_model", "link_status"],
            }

    except Exception as e:
        logger.error(f"Error in llm_lmstudio operation {operation}: {e}", exc_info=True)
        return {"error": f"Operation failed: {e!s}", "operation": operation}


def register_llm_lmstudio_tools(mcp):
    """Register the LM Studio Portmanteau tool with the MCP server."""
    if not FASTMCP_AVAILABLE:
        logger.error("Cannot register LM Studio tools - FastMCP not available")
        return mcp

    _MUTATING = {}

    @mcp.tool(annotations=_MUTATING)
    async def llm_lmstudio_tool(
        operation: str,
        model_path: str | None = None,
        model_name: str | None = None,
        device: str | None = None,
    ) -> dict[str, Any]:
        """LM Studio Portmanteau Tool - Consolidated LM Studio operations.

        This tool consolidates all LM Studio operations into a single interface,
        reducing the number of MCP tools while maintaining full functionality,
        including LM Link peer discovery (Tailscale-powered remote LLM access).

        Use the 'operation' parameter to specify what you want to do:
        - list_models: List all loaded models
        - load_model: Load a model from file path (requires model_path parameter)
        - unload_model: Unload a model (requires model_name parameter)
        - link_status: Show LM Link peers, their loaded models, and link state
        """
        return await llm_lmstudio(
            operation=operation,
            model_path=model_path,
            model_name=model_name,
            device=device,
        )

    logger.info("Registered LM Studio portmanteau tool")
    return mcp
