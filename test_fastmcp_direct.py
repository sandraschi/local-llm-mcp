"""Direct test of FastMCP server."""
import asyncio
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("test_fastmcp")

async def test_fastmcp():
    """Test FastMCP server directly."""
    try:
        from fastmcp import FastMCP
        logger.info("FastMCP imported successfully")
        
        # Create a simple FastMCP instance
        mcp = FastMCP(
            name="Test Server",
            version="0.1.0",
            transport="stdio",
            log_level="DEBUG"
        )
        
        # Add a simple tool
        @mcp.tool()
        async def hello(name: str = "World") -> str:
            """Say hello to someone."""
            return f"Hello, {name}!"
        
        logger.info("Starting FastMCP server...")
        await mcp.start()
        
        try:
            # Keep the server running
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Server shutdown requested")
        finally:
            await mcp.stop()
            
    except Exception as e:
        logger.error(f"Error in test_fastmcp: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        # Configure asyncio for Windows
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Run the test
        asyncio.run(test_fastmcp())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
