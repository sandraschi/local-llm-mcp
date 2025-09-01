"""Main entry point for the LLM MCP server - FIXED for FastMCP 2.12+.

This module initializes and runs the LLM Model Control Protocol server with all available tools.
Includes comprehensive error handling, structured logging, and graceful shutdown.
"""
import asyncio
import json
import logging
import signal
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import structlog
from rich.console import Console
from rich.logging import RichHandler

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Initialize rich console for beautiful output
console = Console()

# Configure structured logging with rotation
def setup_logging() -> structlog.BoundLogger:
    """Set up comprehensive structured logging with file rotation."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Clear any existing handlers
    logging.root.handlers.clear()
    
    # Configure file handler with rotation
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        log_dir / "llm_mcp.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    
    # Configure rich handler for console
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        markup=True
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[file_handler, rich_handler],
        force=True
    )
    
    logger = structlog.get_logger(__name__)
    logger.info("Structured logging initialized", log_file=str(log_dir / "llm_mcp.log"))
    return logger

logger = setup_logging()

try:
    from fastmcp import FastMCP
    from fastmcp.transports import StdioTransport
    from llm_mcp.tools import register_all_tools
    from llm_mcp.state import state_manager
    from llm_mcp.config import Config
except ImportError as e:
    logger.error("Failed to import required modules", error=str(e), traceback=traceback.format_exc())
    console.print(f"[red]Critical import error: {e}[/red]")
    sys.exit(1)

# Global server instance and shutdown flag
server: Optional[FastMCP] = None
shutdown_event = asyncio.Event()

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
            # Cleanup state manager
            if hasattr(state_manager, 'cleanup'):
                await state_manager.cleanup()
            
            # Add any additional cleanup logic here
            await asyncio.sleep(0.1)  # Allow final log writes
            
        except Exception as e:
            logger.error("Error during cleanup", error=str(e))
        
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
        logger.info("Signal received", signal=signum)
        asyncio.create_task(shutdown_handler.shutdown(signum))
    
    if sys.platform == 'win32':
        # Windows signal handling
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    else:
        # Unix-like systems
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, signal_handler)

async def create_mcp_server() -> FastMCP:
    """Create and configure the MCP server with all tools - FIXED for FastMCP 2.12+."""
    try:
        # Load configuration
        config = Config.load()
        logger.info("Configuration loaded", config_path=str(config.config_path))
        
        # Initialize MCP server with FastMCP 2.12+ API
        mcp = FastMCP(
            name="Local LLM MCP Server",
            version="1.0.0"
        )
        
        # Register all tools with error isolation
        logger.info("Registering tools...")
        registration_results = register_all_tools(mcp)
        
        # Log registration results with structured data
        successful_tools = []
        failed_tools = []
        
        for tool_name, status in registration_results.items():
            if status is True:
                successful_tools.append(tool_name)
                logger.info("Tool registered successfully", tool=tool_name)
            else:
                failed_tools.append({"tool": tool_name, "error": str(status)})
                logger.warning("Tool registration failed", tool=tool_name, error=str(status))
        
        logger.info(
            "Tool registration complete",
            successful_count=len(successful_tools),
            failed_count=len(failed_tools),
            successful_tools=successful_tools
        )
        
        # Add health check tool
        @mcp.tool()
        async def health_check() -> Dict[str, Any]:
            """Check the health of the MCP server."""
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "server_version": "1.0.0",
                "models_loaded": len(state_manager.models) if hasattr(state_manager, 'models') else 0,
                "active_sessions": len(state_manager.sessions) if hasattr(state_manager, 'sessions') else 0,
                "registered_tools": len(successful_tools),
                "failed_tools": len(failed_tools),
                "system": {
                    "python": sys.version,
                    "platform": sys.platform,
                    "executable": sys.executable
                }
            }
        
        logger.info("MCP server created successfully")
        return mcp
        
    except Exception as e:
        logger.error("Failed to create MCP server", error=str(e), traceback=traceback.format_exc())
        raise

async def run_server():
    """Run the MCP server with proper FastMCP 2.12+ transport handling."""
    global server
    
    try:
        # Set up signal handlers
        setup_signal_handlers()
        
        # Create the MCP server
        server = await create_mcp_server()
        
        # Create stdio transport (FastMCP 2.12+ pattern)
        transport = StdioTransport()
        
        logger.info("Starting MCP server with stdio transport")
        
        # Run server with transport - FIXED for FastMCP 2.12+
        async with transport:
            await server.run(transport)
            
    except Exception as e:
        logger.error("Fatal server error", error=str(e), traceback=traceback.format_exc())
        raise

async def main():
    """Main entry point with comprehensive error handling."""
    try:
        console.print("[bold green]🚀 Starting Local LLM MCP Server v1.0.0[/bold green]")
        logger.info("Server startup initiated", version="1.0.0")
        
        # Run the server
        server_task = asyncio.create_task(run_server())
        
        # Wait for shutdown signal or server completion
        done, pending = await asyncio.wait(
            [server_task, asyncio.create_task(shutdown_event.wait())],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Handle completed task
        for task in done:
            if task == server_task and not task.cancelled():
                try:
                    await task  # Re-raise any exception
                except Exception as e:
                    logger.error("Server task failed", error=str(e))
                    return 1
        
        logger.info("Server shutdown complete")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        await shutdown_handler.shutdown()
        return 0
    except Exception as e:
        logger.error("Unhandled exception in main", error=str(e), traceback=traceback.format_exc())
        return 1
    finally:
        await shutdown_handler.cleanup()

def cli():
    """Command line interface entry point."""
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console.print("[yellow]\\nShutdown by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        logger.error("CLI fatal error", error=str(e), traceback=traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    cli()
