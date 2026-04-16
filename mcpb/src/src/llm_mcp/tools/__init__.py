"""LLM MCP Tools Package - FIXED for FastMCP 2.12+ and robust registration.

This package contains all the tools for the LLM MCP server with error isolation
and proper dependency management.
"""
import importlib.metadata
import os

import structlog

logger = structlog.get_logger(__name__)

# Updated minimum required versions for 2025 standards
REQUIRED_VERSIONS = {
    'fastmcp': '2.12.0',
    'pydantic': '2.8.0',
    'transformers': '4.44.0',
    'torch': '2.4.0',
    'vllm': '1.0.0',  # vLLM 1.0+ for performance
    'peft': '0.12.0',
    'accelerate': '0.32.0',
}


def check_dependencies() -> dict[str, bool]:
    """Check if all required dependencies are installed and at the correct version."""
    results = {}
    for pkg, min_version in REQUIRED_VERSIONS.items():
        try:
            version = importlib.metadata.version(pkg)
            # Simple version comparison (works for most cases)
            results[pkg] = version >= min_version
            if not results[pkg]:
                logger.warning("Outdated package version", package=pkg, installed=version, required=min_version)
            else:
                logger.debug("Package version OK", package=pkg, version=version)
        except importlib.metadata.PackageNotFoundError:
            results[pkg] = False
            logger.warning("Package not installed", package=pkg)
    return results


# Check dependencies on import
dependency_status = check_dependencies()


def safe_import_tool_module(module_name: str, register_func: str):
    """Safely import a tool module with error isolation."""
    try:
        module = __import__(f"llm_mcp.tools.{module_name}", fromlist=[register_func])
        return getattr(module, register_func)
    except ImportError as e:
        logger.warning("Failed to import tool module", module=module_name, error=str(e))
        return None
    except AttributeError as e:
        logger.warning("Tool registration function not found",
                      module=module_name, function=register_func, error=str(e))
        return None


def register_all_tools(mcp):
    """Register all available tools with the MCP server with error isolation.

    PORTMANTEAU ARCHITECTURE:
    This server now uses a SOTA portmanteau architecture following Advanced Memory MCP patterns.
    Instead of 30+ individual tools, we provide 5 consolidated portmanteau tools:

    - llm_health: Health, monitoring, system, and server management operations
    - llm_models: Model management, registration, and provider operations
    - llm_generation: Text generation, chat completion, and embedding operations
    - llm_multimodal: Image analysis, generation, and comparison operations
    - llm_finetuning: LoRA, Sparse, and DoRA fine-tuning operations

    Legacy individual tools can be enabled with LLM_MCP_ENABLE_LEGACY_TOOLS=true for migration testing.

    Args:
        mcp: The MCP server instance

    Returns:
        The MCP server instance with all tools registered
    """
    registration_results = {}

    # PORTMANTEAU TOOLS - SOTA consolidated interface
    # Following Advanced Memory MCP pattern to reduce tool count and improve UX
    portmanteau_tools = [
        ("portmanteau_health", "register_llm_health_tools"),
        ("portmanteau_models", "register_llm_models_tools"),
        ("portmanteau_generation", "register_llm_generation_tools"),
        ("portmanteau_multimodal", "register_llm_multimodal_tools"),
        ("portmanteau_finetuning", "register_llm_finetuning_tools"),
    ]

    # Log portmanteau tool registration progress
    logger.info("Registering SOTA portmanteau tools...")
    for module_name, func_name in portmanteau_tools:
        try:
            register_func = safe_import_tool_module(module_name, func_name)
            if register_func:
                mcp = register_func(mcp)  # Update mcp with the returned instance
                registration_results[func_name] = True
                logger.info(f"Successfully registered portmanteau tool: {func_name}")
            else:
                registration_results[func_name] = "Import failed"
                logger.warning(f"Failed to import portmanteau tool: {func_name}")
        except Exception as e:
            error_msg = f"Error registering portmanteau tool {func_name}: {e!s}"
            logger.error(error_msg, exc_info=True)
            registration_results[func_name] = error_msg

    # LEGACY INDIVIDUAL TOOLS - kept for backward compatibility but marked as deprecated
    # These will be removed in a future version after migration period
    legacy_core_tools = [
        ("help_tools", "register_help_tools"),
        ("system_tools", "register_system_tools"),
        ("monitoring_tools", "register_monitoring_tools"),
    ]

    # Register legacy tools only if explicitly requested (for migration testing)
    if os.getenv("LLM_MCP_ENABLE_LEGACY_TOOLS", "").lower() in ("true", "1", "yes"):
        logger.warning("Registering legacy individual tools - consider migrating to portmanteau interface")
        for module_name, func_name in legacy_core_tools:
            try:
                register_func = safe_import_tool_module(module_name, func_name)
                if register_func:
                    mcp = register_func(mcp)
                    registration_results[f"legacy_{func_name}"] = True
                    logger.info(f"Registered legacy tool: {func_name}")
                else:
                    registration_results[f"legacy_{func_name}"] = "Import failed"
            except Exception as e:
                error_msg = f"Error registering legacy tool {func_name}: {e!s}"
                logger.error(error_msg)
                registration_results[f"legacy_{func_name}"] = error_msg

    # Model management tools (require basic ML dependencies)
    ml_basic_tools = [
        ("model_tools", "register_model_tools"),
        ("generation_tools", "register_generation_tools"),
        ("model_management_tools", "register_model_management_tools"),
    ]

    # Check for torch and transformers (allow any installed version)
    torch_installed = False
    transformers_installed = False

    try:
        import torch
        torch_installed = True
    except ImportError:
        pass

    try:
        import transformers
        transformers_installed = True
    except ImportError:
        pass

    if torch_installed and transformers_installed:
        # Log ML tool registration progress
        logger.info("Registering ML tools...")
        for module_name, func_name in ml_basic_tools:
            try:
                register_func = safe_import_tool_module(module_name, func_name)
                if register_func:
                    mcp = register_func(mcp)  # Update mcp with the returned instance
                    registration_results[func_name] = True
                    logger.info(f"Successfully registered {func_name}")
                else:
                    registration_results[func_name] = "Import failed"
                    logger.warning(f"Failed to import {func_name}")
            except Exception as e:
                # Use safe error message to avoid Unicode encoding issues
                error_msg = f"Error registering {func_name}: {e!s}"
                try:
                    logger.error(error_msg, exc_info=True)
                except (UnicodeEncodeError, UnicodeDecodeError):
                    # Fallback for Unicode encoding issues
                    logger.error(f"Error registering {func_name}: {type(e).__name__}")
                registration_results[func_name] = error_msg
    else:
        missing_deps = []
        if not dependency_status.get('torch', False):
            missing_deps.append('torch')
        if not dependency_status.get('transformers', False):
            missing_deps.append('transformers')

        warning_msg = f"Skipping ML tools - missing dependencies: {', '.join(missing_deps)}"
        logger.warning(warning_msg)
        for _, func_name in ml_basic_tools:
            registration_results[func_name] = warning_msg

    # Advanced tools with specific requirements (non-portmanteau)
    # Note: LoRA, multimodal, and other fine-tuning tools are now consolidated into portmanteau_finetuning
    advanced_tools = [
        # vLLM 1.0+ tools - high priority (not yet portmanteau-ized)
        {
            "module": "vllm_tools",
            "function": "register_vllm_tools",
            "deps": ["vllm", "torch"],
            "description": "vLLM 1.0+ high-performance inference"
        },

        # Additional advanced tools (non-consolidated)
        {
            "module": "moe_tools",
            "function": "register_moe_tools",
            "deps": ["torch"],
            "description": "Mixture of Experts models"
        },

        # UI and visualization tools
        {
            "module": "gradio_tools",
            "function": "register_gradio_tools",
            "deps": ["gradio"],
            "description": "Gradio web interface tools"
        },

        # Legacy fine-tuning tools (available via LLM_MCP_ENABLE_LEGACY_TOOLS=true)
        # These are now consolidated into portmanteau_finetuning but kept for migration
        {
            "module": "unsloth_tools",
            "function": "register_unsloth_tools",
            "deps": ["torch"],
            "description": "Unsloth efficient fine-tuning (LEGACY - use portmanteau_finetuning)"
        },

        {
            "module": "qloraevolved_tools",
            "function": "register_qloraevolved_tools",
            "deps": ["peft", "torch"],
            "description": "QLoRA evolved training methods (LEGACY - use portmanteau_finetuning)"
        },
    ]

    # Log advanced tool registration progress
    logger.info("Registering advanced tools...")
    for tool_config in advanced_tools:
        module_name = tool_config["module"]
        func_name = tool_config["function"]
        required_deps = tool_config["deps"]
        description = tool_config["description"]

        # Check if dependencies are available
        missing_deps = [dep for dep in required_deps if not dependency_status.get(dep, False)]

        if missing_deps:
            registration_results[func_name] = f"Missing dependencies: {', '.join(missing_deps)}"
            logger.info(f"Skipping advanced tool {func_name} - missing dependencies: {', '.join(missing_deps)}")
            continue

        # Try to register the tool with error isolation
        try:
            register_func = safe_import_tool_module(module_name, func_name)
            if register_func:
                result = register_func(mcp)

                # Handle different return types from registration functions
                if isinstance(result, dict):
                    if result.get("vllm_available", True):  # vLLM-specific check
                        registration_results[func_name] = True
                        logger.info(f"Advanced tool registered: {func_name}")
                    else:
                        registration_results[func_name] = result.get("error", "Registration returned false")
                        logger.warning("Advanced tool registration failed", tool=func_name, result=result)
                else:
                    # Assume success if function returns mcp instance or None
                    registration_results[func_name] = True
                    logger.info(f"Advanced tool registered: {func_name}")

            else:
                registration_results[func_name] = "Import failed"

        except Exception as e:
            logger.error("Failed to register advanced tool",
                        tool=func_name,
                        description=description,
                        error=str(e))
            registration_results[func_name] = str(e)

    # Log summary
    [k for k, v in registration_results.items() if v is True]
    failed_tools = {k: v for k, v in registration_results.items() if v is not True}

    logger.info("Tool registration complete")

    if failed_tools:
        logger.warning(f"Some tools failed to register: {failed_tools}")

    return mcp


# Tool registration functions for backward compatibility
def register_help_tools(mcp):
    """Stub for help tools registration - implemented in help_tools.py"""
    pass


def register_model_tools(mcp):
    """Stub for model tools registration - implemented in model_tools.py"""
    pass


def register_generation_tools(mcp):
    """Stub for generation tools registration - implemented in generation_tools.py"""
    pass


def register_monitoring_tools(mcp):
    """Stub for monitoring tools registration - implemented in monitoring_tools.py"""
    pass


def register_system_tools(mcp):
    """Stub for system tools registration - implemented in system_tools.py"""
    pass


def register_model_management_tools(mcp):
    """Stub for model management tools registration - implemented in model_management_tools.py"""
    pass


def register_vllm_tools(mcp):
    """Stub for vLLM tools registration - implemented in vllm_tools.py"""
    pass


def register_multimodal_tools(mcp):
    """Stub for multimodal tools registration - implemented in multimodal_tools.py"""
    pass


def register_lora_tools(mcp):
    """Stub for LoRA tools registration - implemented in lora_tools.py"""
    pass


def register_gradio_tools(mcp):
    """Stub for Gradio tools registration - implemented in gradio_tools.py"""
    pass


def register_unsloth_tools(mcp):
    """Stub for Unsloth tools registration - implemented in unsloth_tools.py"""
    pass


def register_qloraevolved_tools(mcp):
    """Stub for QLoRA evolved tools registration - implemented in qloraevolved_tools.py"""
    pass


def register_dora_tools(mcp):
    """Stub for DoRA tools registration - implemented in dora_tools.py"""
    pass


def register_sparse_tools(mcp):
    """Stub for sparse tools registration - implemented in sparse_tools.py"""
    pass


def register_moe_tools(mcp):
    """Stub for MoE tools registration - implemented in moe_tools.py"""
    pass


__all__ = [
    'check_dependencies',
    'register_all_tools',
    'register_dora_tools',
    'register_generation_tools',
    'register_gradio_tools',
    'register_help_tools',
    'register_lora_tools',
    'register_model_management_tools',
    'register_model_tools',
    'register_moe_tools',
    'register_monitoring_tools',
    'register_multimodal_tools',
    'register_qloraevolved_tools',
    'register_sparse_tools',
    'register_system_tools',
    'register_unsloth_tools',
    'register_vllm_tools',
]
