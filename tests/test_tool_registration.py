"""Test tool registration for the LLM MCP server."""
import asyncio
import logging
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

from fastmcp import FastMCP
from llm_mcp.tools import check_dependencies, register_all_tools

async def test_tool_registration():
    """Test that all tools are properly registered."""
    # Check dependencies first
    deps = check_dependencies()
    missing_critical = [k for k, v in deps.items() 
                       if not v and k in {'fastmcp', 'torch', 'transformers'}]
    
    if missing_critical:
        logger.warning(f"Missing critical dependencies: {', '.join(missing_critical)}")
        logger.warning("Some features may not work as expected")
    
    # Initialize MCP server
    mcp = FastMCP(
        name="LLM MCP Test",
        version="1.0.0",
        description="Test server for tool registration"
    )
    
    # Register all tools
    logger.info("Registering all tools...")
    registration_results = register_all_tools(mcp)
    
    # Print registration results
    logger.info("\nTool Registration Results:")
    logger.info("=" * 50)
    
    success_count = 0
    for tool, status in registration_results.items():
        status_str = "✓" if status is True else f"✗ ({status})"
        logger.info(f"{tool}: {status_str}")
        if status is True:
            success_count += 1
    
    total_tools = len(registration_results)
    success_rate = (success_count / total_tools) * 100
    
    logger.info("=" * 50)
    logger.info(f"Successfully registered {success_count}/{total_tools} tools ({success_rate:.1f}%)")
    
    # List all registered tools
    logger.info("\nRegistered Tools:")
    logger.info("=" * 50)
    for tool_name in mcp.tools:
        logger.info(f"- {tool_name}")
    
    return registration_results

if __name__ == "__main__":
    asyncio.run(test_tool_registration())
