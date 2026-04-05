"""Simple test script for FastMCP."""
import sys
import logging
import asyncio
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("test_fastmcp")

async def test_fastmcp():
    """Test FastMCP basic functionality."""
    try:
        # Import FastMCP
        logger.info("Importing FastMCP...")
        from fastmcp import FastMCP
        
        # Print FastMCP info
        logger.info(f"FastMCP location: {FastMCP.__module__}")
        logger.info(f"FastMCP attributes: {[attr for attr in dir(FastMCP) if not attr.startswith('_')]}")
        
        # Create a simple FastMCP instance
        logger.info("Creating FastMCP instance...")
        mcp = FastMCP(
            name="Test Server",
            version="0.1.0"
        )
        
        # Add a simple tool
        @mcp.tool()
        async def hello(name: str = "World") -> str:
            """Say hello to someone."""
            return f"Hello, {name}!"
        
        # List registered tools
        logger.info(f"Registered tools: {[t.name for t in mcp.tools]}")
        
        # Test the tool
        logger.info("Testing hello tool...")
        result = await hello("FastMCP Tester")
        logger.info(f"Tool result: {result}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in test_fastmcp: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    try:
        # Configure asyncio for Windows
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Run the test
        success = asyncio.run(test_fastmcp())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
