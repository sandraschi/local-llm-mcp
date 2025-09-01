"""Test tool registration for the LLM MCP server."""
import asyncio
import logging
import sys
import traceback
import importlib
from pathlib import Path

# Add the src directory to the path
src_path = str(Path(__file__).parent / "src")
sys.path.insert(0, src_path)
print(f"Added to Python path: {src_path}")

# Debug: Print Python path and verify src directory
print("\nPython path:")
for p in sys.path:
    print(f"- {p}")

# Debug: List files in src directory
print("\nFiles in src directory:")
try:
    src_dir = Path(__file__).parent / "src"
    for f in src_dir.rglob("*.py"):
        print(f"- {f.relative_to(src_dir.parent)}")
except Exception as e:
    print(f"Error listing src directory: {e}")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more detailed output
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Debug: Try to import FastMCP
try:
    print("\nAttempting to import FastMCP...")
    fastmcp_spec = importlib.util.find_spec("fastmcp")
    if fastmcp_spec is None:
        print("Error: fastmcp module not found in Python path")
    else:
        print(f"Found fastmcp at: {fastmcp_spec.origin}")
        from fastmcp import FastMCP
        print("Successfully imported FastMCP")
except Exception as e:
    print(f"Error importing FastMCP: {e}")
    print(traceback.format_exc())
    sys.exit(1)

# Debug: Try to import llm_mcp
try:
    print("\nAttempting to import llm_mcp...")
    llm_mcp_spec = importlib.util.find_spec("llm_mcp")
    if llm_mcp_spec is None:
        print("Error: llm_mcp package not found in Python path")
    else:
        print(f"Found llm_mcp at: {llm_mcp_spec.origin}")
        
        # Try to import from tools
        print("\nAttempting to import from llm_mcp.tools...")
        try:
            from llm_mcp.tools import check_dependencies, register_all_tools
            print("Successfully imported from llm_mcp.tools")
        except ImportError as e:
            print(f"Error importing from llm_mcp.tools: {e}")
            print(traceback.format_exc())
            
            # Try to import tools directly
            print("\nTrying to import tools directly...")
            try:
                import llm_mcp.tools
                print(f"Imported llm_mcp.tools from {llm_mcp.tools.__file__}")
                check_dependencies = llm_mcp.tools.check_dependencies
                register_all_tools = llm_mcp.tools.register_all_tools
                print("Assigned functions from module")
            except Exception as e2:
                print(f"Failed to import tools directly: {e2}")
                print(traceback.format_exc())
                sys.exit(1)
    
except Exception as e:
    print(f"Error importing llm_mcp: {e}")
    print(traceback.format_exc())
    sys.exit(1)

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
    print("\nTool Registration Results:")
    print("=" * 50)
    
    success_count = 0
    for tool, status in registration_results.items():
        status_str = "✓" if status is True else f"✗ ({status})"
        print(f"{tool}: {status_str}")
        if status is True:
            success_count += 1
    
    total_tools = len(registration_results)
    success_rate = (success_count / total_tools) * 100
    
    print("=" * 50)
    print(f"Successfully registered {success_count}/{total_tools} tools ({success_rate:.1f}%)")
    
    # List all registered tools
    print("\nRegistered Tools:")
    print("=" * 50)
    for tool_name in mcp.tools:
        print(f"- {tool_name}")
    
    return registration_results

if __name__ == "__main__":
    try:
        print("Starting tool registration test...")
        asyncio.run(test_tool_registration())
    except Exception as e:
        logger.error(f"Test failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
