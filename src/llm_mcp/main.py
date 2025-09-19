"""Main entry point for the LLM MCP server - FastMCP 2.12+ compliant.

This module initializes and runs the LLM Model Control Protocol server with all available tools.
Includes comprehensive error handling, structured logging, and graceful shutdown.
"""
import asyncio
import json
import logging
import signal
import sys
import traceback
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Awaitable

# Suppress warnings to prevent them from interfering with JSON-RPC communication
warnings.filterwarnings('ignore')

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our centralized logging configuration
from llm_mcp.utils.logging import LoggingConfig, get_logger

# Initialize logging with reduced verbosity for MCP stdio transport
LoggingConfig.initialize(log_level="WARNING")  # Reduce log verbosity
logger = get_logger(__name__)

# Import FastMCP after logging is configured
# Try to import FastMCP with version check
try:
    import pkg_resources
    from fastmcp import FastMCP, __version__ as fastmcp_version_installed
    from fastmcp.tools import Tool
    
    # Check if the installed version meets our requirements
    required_version = pkg_resources.parse_version("2.12.0")
    installed_version = pkg_resources.parse_version(fastmcp_version_installed)
    
    if installed_version < required_version:
        raise ImportError(f"FastMCP version {required_version} or higher is required. Installed version: {installed_version}")
    
    FASTMCP_AVAILABLE = True
    logger.info(f"Using FastMCP version: {fastmcp_version_installed}")
    
except ImportError as e:
    logger.error(f"FastMCP import error: {str(e)}")
    logger.error("Please install the required version with: pip install 'fastmcp>=2.12.0'")
    FASTMCP_AVAILABLE = False
except Exception as e:
    logger.error(f"Error checking FastMCP version: {str(e)}")
    FASTMCP_AVAILABLE = False
    sys.exit(1)

# Import local modules
from llm_mcp.services.provider_factory import ProviderFactory
from llm_mcp.services.model_manager import ModelManager
from llm_mcp.tools import register_all_tools
from llm_mcp.config import Config

# Global state
shutdown_event = asyncio.Event()
server = None

class GracefulShutdown:
    """Handle graceful shutdown with cleanup."""
    
    def __init__(self):
        self.shutdown_started = False
    
    async def cleanup(self):
        """Clean up resources."""
        if self.shutdown_started:
            return
        
        self.shutdown_started = True
        logger.info("Starting graceful shutdown cleanup")
        
        try:
            # Cleanup state manager if it exists
            if 'state_manager' in globals() and hasattr(state_manager, 'cleanup'):
                await state_manager.cleanup()
            
            # Add any additional cleanup logic here
            await asyncio.sleep(0.1)  # Allow final log writes
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
        
        logger.info("Cleanup complete")
    
    async def shutdown(self, signal_received=None):
        """Handle graceful shutdown."""
        if signal_received:
            logger.info("Shutdown signal received", signal=signal_received.name if hasattr(signal_received, 'name') else str(signal_received))
        else:
            logger.info("Shutdown requested")
        
        await self.cleanup()
        shutdown_event.set()

shutdown_handler = GracefulShutdown()

def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Signal {signum} received")
        asyncio.create_task(shutdown_handler.shutdown(signum))
    
    if sys.platform == 'win32':
        # Windows signal handling
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    else:
        # Unix-like systems
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, signal_handler)

# Global state manager
state_manager = None

# Global server instance
server = None

async def create_mcp_server_sync() -> Optional[FastMCP]:
    """Create and configure the MCP server with all tools for FastMCP 2.12+ (synchronous version).
    
    Returns:
        FastMCP: Configured MCP server instance or None if initialization fails
    """
    try:
        # Load configuration
        config = Config.load()
        logger.info(f"Configuration loaded from {config.config_path}")
        
        # Initialize MCP server with FastMCP 2.12+ API
        try:
            mcp = FastMCP(
                name="LLM MCP Server",
                version="1.0.0"
            )
            
            logger.info("MCP server instance created successfully")
            
            # Add health check tool
            @mcp.tool(
                name="health_check",
                description="Check the health of the MCP server and list available tools."
            )
            async def health_check(verbose: bool = False) -> Dict[str, Any]:
                """Check the health of the MCP server."""
                response = {
                    "status": "healthy",
                    "timestamp": datetime.utcnow().isoformat(),
                    "server_version": "1.0.0",
                    "registered_tools": list((await mcp.get_tools()).keys())
                }
                
                if verbose:
                    response.update({
                        "system": {
                            "python": sys.version,
                            "platform": sys.platform,
                            "executable": sys.executable
                        }
                    })
                
                return response

            # Register all tools with error isolation
            try:
                # Register all tools (suppress logging during registration)
                mcp = register_all_tools(mcp)
                
                # Log registered tools
                tools = await mcp.get_tools()
                tool_count = len([name for name in tools.keys() if not name.startswith('_')])
                logger.warning(f"MCP server ready with {tool_count} tools")
                
                
                logger.info("MCP server initialized successfully")
                return mcp
                
            except Exception as e:
                logger.error(f"Failed to register tools: {str(e)}", exc_info=True)
                logger.error("This might be due to missing dependencies or configuration issues")
                # Try to return the server even if some tools failed to register
                return mcp if 'mcp' in locals() else None
                
        except Exception as e:
            logger.error(f"Failed to initialize MCP server: {str(e)}", exc_info=True)
            return None
            
    except Exception as e:
        logger.error(f"Failed to create MCP server: {str(e)}", exc_info=True)
        return None


async def run_server():
    """Run the MCP server with stdio transport for FastMCP 2.12+."""
    global server
    
    try:
        # Set up signal handlers
        setup_signal_handlers()
        
        # Create the MCP server
        server = await create_mcp_server_sync()
        
        if server is None:
            logger.error("Failed to create MCP server")
            return 1
        
        logger.info("Starting MCP server with stdio transport")
        
        # Run server with stdio transport (handled by FastMCP 2.12+)
        await server.run_stdio_async()
        
        logger.info("MCP server stopped")
        return 0
            
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
        return 0
    except Exception as e:
        logger.critical(f"Fatal server error: {str(e)}", exc_info=True)
        return 1

async def main() -> int:
    """Main entry point for the LLM MCP server."""
    try:
        if not FASTMCP_AVAILABLE:
            logger.error("Cannot start server: FastMCP is not available")
            logger.error("Please ensure you have installed the package in development mode with: pip install -e .")
            logger.error("Also verify that the package is in your PYTHONPATH")
            return 1
        
        # Log Python path and environment for debugging
        logger.info(f"Python path: {sys.path}")
        logger.info(f"Python executable: {sys.executable}")
        
        # Run the server directly
        return await run_server()
            
    except KeyboardInterrupt:
        logger.info("Main task cancelled by user")
        return 0
    except Exception as e:
        logger.critical(f"Fatal error in main: {str(e)}", exc_info=True)
        return 1
    finally:
        logger.info("Server shutdown complete")

def cli():
    """Command line interface entry point."""
    try:
        logger.info("Starting LLM MCP Server CLI")
        
        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            shutdown_event.set()
            
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal_handler)
        
        # Run the main async function
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.info("Shutdown by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    cli()
