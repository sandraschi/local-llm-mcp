"""Minimal FastMCP server test."""
import asyncio
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("test_server")

async def main():
    """Run a minimal FastMCP server."""
    try:
        # Import FastMCP
        from fastmcp import FastMCP
        logger.info("Successfully imported FastMCP")
        
        # Create FastMCP instance
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
        
        # Start the server
        logger.info("Starting server...")
        await mcp.start()
        
        # Keep the server running
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Server shutdown requested")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1
    finally:
        if 'mcp' in locals() and hasattr(mcp, 'stop'):
            await mcp.stop()

if __name__ == "__main__":
    try:
        # Configure asyncio for Windows
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Run the server
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
