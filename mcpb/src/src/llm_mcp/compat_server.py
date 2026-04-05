"""Compatibility server for FastMCP."""
import asyncio
import logging
import sys
import signal
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Awaitable

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure basic logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("compat_server")

class CompatServer:
    """Compatibility server for FastMCP."""
    
    def __init__(self):
        """Initialize the compatibility server."""
        self.server = None
        self.shutdown_event = asyncio.Event()
        self.tools: List[Callable[..., Awaitable[Any]]] = []
    
    async def health_check(self, verbose: bool = False) -> Dict[str, Any]:
        """Basic health check endpoint."""
        return {
            "status": "healthy",
            "version": "0.1.0",
            "server": "compat_server",
            "timestamp": asyncio.get_event_loop().time()
        }
    
    def tool(self, func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        """Decorator to register a tool."""
        self.tools.append(func)
        return func
    
    async def start(self):
        """Start the compatibility server."""
        try:
            # Import FastMCP here to get better error messages
            try:
                from fastmcp import FastMCP
                logger.info("Successfully imported FastMCP")
            except ImportError as e:
                logger.error(f"Failed to import FastMCP: {e}")
                logger.error("Please install with: pip install 'fastmcp>=2.12.0'")
                sys.exit(1)
            
            logger.info("Creating FastMCP instance...")
            
            # Create FastMCP instance with minimal required parameters
            self.server = FastMCP(
                name="LLM MCP Server",
                version="0.1.0"
            )
            
            # Register health check
            logger.info("Registering health check tool...")
            self.tool(self.health_check)
            
            # Log registered tools
            logger.info(f"Registered tools: {[f.__name__ for f in self.tools]}")
            
            # Start the server
            logger.info("Starting MCP server...")
            
            try:
                if hasattr(self.server, 'start'):
                    await self.server.start()
                    logger.info("Server started successfully, waiting for connections...")
                    
                    # Keep the server running
                    while not self.shutdown_event.is_set():
                        await asyncio.sleep(0.1)
                else:
                    logger.error("FastMCP instance has no 'start' method")
                    return
                    
            except asyncio.CancelledError:
                logger.info("Server shutdown requested")
            except Exception as e:
                logger.error(f"Server error: {e}", exc_info=True)
                raise
                
        except Exception as e:
            logger.error(f"Failed to start server: {e}", exc_info=True)
            raise
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the server."""
        if self.server is not None:
            logger.info("Stopping server...")
            if hasattr(self.server, 'stop'):
                await self.server.stop()
            self.server = None
        logger.info("Server stopped")

def setup_signal_handlers(server):
    """Set up signal handlers for graceful shutdown."""
    def signal_handler():
        logger.info("Shutdown signal received")
        server.shutdown_event.set()
    
    # Windows-compatible signal handling
    if sys.platform == 'win32':
        # On Windows, we can only handle these signals
        for sig in (signal.SIGINT,):
            try:
                signal.signal(sig, lambda s, f: signal_handler())
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Failed to set signal handler for {sig}: {e}")
    else:
        # On Unix-like systems, we can handle more signals
        try:
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, signal_handler)
        except (NotImplementedError, RuntimeError) as e:
            logger.warning(f"Could not set up signal handlers: {e}")

async def main():
    """Run the compatibility server."""
    logger.info("=== Starting LLM MCP Server ===")
    
    # Log Python and system info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    logger.info(f"Working directory: {Path.cwd()}")
    
    server = CompatServer()
    setup_signal_handlers(server)
    
    try:
        await server.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
    finally:
        await server.stop()
    
    return 0

if __name__ == "__main__":
    # Configure asyncio for Windows
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
