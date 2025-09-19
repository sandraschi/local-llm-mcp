"""LLM MCP Tools Package - FIXED for FastMCP 2.12+ and robust registration.

This package contains all the tools for the LLM MCP server with error isolation
and proper dependency management.
"""
import importlib.metadata
import logging
from typing import Optional, Dict, Any, List
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

def check_dependencies() -> Dict[str, bool]:
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
    
    Args:
        mcp: The MCP server instance
        
    Returns:
        The MCP server instance with all tools registered
    """
    registration_results = {}
    
    # Core tools (always available) - these should work without heavy dependencies
    core_tools = [
        ("help_tools", "register_help_tools"),
        ("system_tools", "register_system_tools"),
        ("monitoring_tools", "register_monitoring_tools"),
    ]
    
    # Suppress verbose logging during tool registration
    for module_name, func_name in core_tools:
        try:
            register_func = safe_import_tool_module(module_name, func_name)
            if register_func:
                mcp = register_func(mcp)  # Update mcp with the returned instance
                registration_results[func_name] = True
                # Successfully registered (suppress logging)
            else:
                registration_results[func_name] = "Import failed"
                logger.warning(f"Failed to import {func_name}")
        except Exception as e:
            error_msg = f"Error registering {func_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            registration_results[func_name] = error_msg
    
    # Model management tools (require basic ML dependencies)
    ml_basic_tools = [
        ("model_tools", "register_model_tools"),
        ("generation_tools", "register_generation_tools"),
        ("model_management_tools", "register_model_management_tools"),
    ]
    
    if dependency_status.get('torch', False) and dependency_status.get('transformers', False):
        logger.info("Registering basic ML tools")
        for module_name, func_name in ml_basic_tools:
            try:
                register_func = safe_import_tool_module(module_name, func_name)
                if register_func:
                    mcp = register_func(mcp)  # Update mcp with the returned instance
                    registration_results[func_name] = True
                    # Successfully registered (suppress logging)
                else:
                    registration_results[func_name] = "Import failed"
                    logger.warning(f"Failed to import {func_name}")
            except Exception as e:
                # Use safe error message to avoid Unicode encoding issues
                error_msg = f"Error registering {func_name}: {str(e)}"
                try:
                    logger.error(error_msg, exc_info=True)
                except UnicodeEncodeError:
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
    
    # Advanced tools with specific requirements
    advanced_tools = [
        # vLLM 1.0+ tools - high priority
        {
            "module": "vllm_tools",
            "function": "register_vllm_tools", 
            "deps": ["vllm", "torch"],
            "description": "vLLM 1.0+ high-performance inference"
        },
        # Training and fine-tuning
        {
            "module": "lora_tools",
            "function": "register_lora_tools",
            "deps": ["peft", "torch"],
            "description": "LoRA parameter-efficient fine-tuning"
        },
        
        # Multimodal capabilities  
        {
            "module": "multimodal_tools",
            "function": "register_multimodal_tools",
            "deps": ["torch", "transformers"],
            "description": "Vision and multimodal model support"
        },
        
        # Advanced training methods
        {
            "module": "unsloth_tools",
            "function": "register_unsloth_tools", 
            "deps": ["torch"],
            "description": "Unsloth efficient fine-tuning"
        },
        
        {
            "module": "qloraevolved_tools",
            "function": "register_qloraevolved_tools",
            "deps": ["peft", "torch"],
            "description": "QLoRA evolved training methods"
        },
        
        # Specialized architectures
        {
            "module": "sparse_tools",
            "function": "register_sparse_tools",
            "deps": ["torch"],
            "description": "Sparse model optimization"
        },
        
        {
            "module": "moe_tools", 
            "function": "register_moe_tools",
            "deps": ["torch"],
            "description": "Mixture of Experts models"
        },
        
        # DoRA (Weight-Decomposed Low-Rank Adaptation)
        {
            "module": "dora_tools",
            "function": "register_dora_tools",
            "deps": ["peft", "torch"],
            "description": "DoRA weight decomposition"
        },
        
        # UI and visualization
        {
            "module": "gradio_tools",
            "function": "register_gradio_tools",
            "deps": ["gradio"],
            "description": "Gradio web interface tools"
        },
    ]
    
    logger.info("Registering advanced tools", count=len(advanced_tools))
    for tool_config in advanced_tools:
        module_name = tool_config["module"]
        func_name = tool_config["function"]
        required_deps = tool_config["deps"]
        description = tool_config["description"]
        
        # Check if dependencies are available
        missing_deps = [dep for dep in required_deps if not dependency_status.get(dep, False)]
        
        if missing_deps:
            registration_results[func_name] = f"Missing dependencies: {', '.join(missing_deps)}"
            logger.info("Skipping advanced tool", 
                       tool=func_name, 
                       missing=missing_deps,
                       description=description)
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
                        logger.info("Advanced tool registered", tool=func_name, description=description)
                    else:
                        registration_results[func_name] = result.get("error", "Registration returned false")
                        logger.warning("Advanced tool registration failed", tool=func_name, result=result)
                else:
                    # Assume success if function returns mcp instance or None
                    registration_results[func_name] = True
                    logger.info("Advanced tool registered", tool=func_name, description=description)
                    
            else:
                registration_results[func_name] = "Import failed"
                
        except Exception as e:
            logger.error("Failed to register advanced tool", 
                        tool=func_name, 
                        description=description,
                        error=str(e))
            registration_results[func_name] = str(e)
    
    # Log summary
    successful_tools = [k for k, v in registration_results.items() if v is True]
    failed_tools = {k: v for k, v in registration_results.items() if v is not True}
    
    logger.info("Tool registration complete",
                total_tools=len(registration_results),
                successful=len(successful_tools),
                failed=len(failed_tools))
    
    if failed_tools:
        logger.warning("Some tools failed to register", failed_tools=failed_tools)
    
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
    'register_help_tools',
    'register_model_tools',
    'register_generation_tools',
    'register_monitoring_tools',
    'register_system_tools',
    'register_model_management_tools',
    'register_vllm_tools',
    'register_multimodal_tools',
    'register_lora_tools',
    'register_gradio_tools',
    'register_unsloth_tools',
    'register_qloraevolved_tools',
    'register_dora_tools',
    'register_sparse_tools',
    'register_moe_tools',
    'register_all_tools',
    'check_dependencies',
]
