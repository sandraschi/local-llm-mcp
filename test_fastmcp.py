"""
Test script to verify FastMCP installation and basic functionality.
Run this with: python test_fastmcp.py
"""
import sys
import asyncio
import logging
from pathlib import Path

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fastmcp_test")

# Add project root to Python path
project_root = str(Path(__file__).parent.resolve())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def check_imports():
    """Check if FastMCP and its dependencies can be imported."""
    try:
        import pkg_resources
        from fastmcp import FastMCP, __version__ as fastmcp_version
        
        logger.info(f"Successfully imported FastMCP version: {fastmcp_version}")
        
        # Check version
        required_version = pkg_resources.parse_version("2.12.0")
        installed_version = pkg_resources.parse_version(fastmcp_version)
        
        if installed_version < required_version:
            logger.error(f"FastMCP version {required_version} or higher is required. Installed: {installed_version}")
            return False
            
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        logger.error("Please install FastMCP with: pip install 'fastmcp>=2.12.0'")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return False

async def test_mcp_server():
    """Test creating a basic MCP server instance."""
    try:
        from fastmcp import FastMCP
        
        logger.info("Creating test MCP server...")
        mcp = FastMCP(
            name="Test MCP Server",
            version="1.0.0",
            description="Test server for FastMCP verification"
        )
        
        # Add a simple tool
        @mcp.tool
        async def test_tool(name: str = "World") -> str:
            """A simple test tool that greets the user."""
            return f"Hello, {name}!"
        
        logger.info("Test MCP server created successfully")
        logger.info(f"Registered tools: {[t.name for t in mcp.tools]}")
        
        # Test the tool
        result = await test_tool("FastMCP Tester")
        logger.info(f"Test tool result: {result}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing MCP server: {str(e)}", exc_info=True)
        return False

async def main():
    """Main test function."""
    logger.info("Starting FastMCP test...")
    
    # Check imports
    if not check_imports():
        return 1
    
    # Test MCP server
    if not await test_mcp_server():
        return 1
    
    logger.info("All tests completed successfully!")
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        sys.exit(1)
