"""Inspect the FastMCP module and its attributes."""
import sys
import inspect
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("inspect_fastmcp")

def inspect_module(module_name):
    """Inspect a module and print its attributes and methods."""
    try:
        logger.info(f"Importing {module_name}...")
        module = __import__(module_name)
        
        # Get the module's attributes
        attrs = [attr for attr in dir(module) if not attr.startswith('_')]
        logger.info(f"\n=== {module_name} attributes ===")
        logger.info(f"Attributes: {', '.join(attrs)}\n")
        
        # Check if FastMCP class exists
        if hasattr(module, 'FastMCP'):
            fastmcp_class = getattr(module, 'FastMCP')
            logger.info("=== FastMCP Class ===")
            logger.info(f"Module: {fastmcp_class.__module__}")
            
            # Get class attributes
            class_attrs = [attr for attr in dir(fastmcp_class) if not attr.startswith('_')]
            logger.info(f"Class attributes: {', '.join(class_attrs)}")
            
            # Get class methods
            methods = [
                name for name, member in inspect.getmembers(fastmcp_class, inspect.isfunction)
                if not name.startswith('_')
            ]
            logger.info(f"Methods: {', '.join(methods)}")
            
            # Get constructor signature
            try:
                sig = inspect.signature(fastmcp_class.__init__)
                logger.info(f"Constructor signature: {sig}")
            except (TypeError, ValueError) as e:
                logger.warning(f"Could not get constructor signature: {e}")
            
            # Try to create an instance
            try:
                logger.info("\nAttempting to create FastMCP instance...")
                instance = fastmcp_class(
                    name="Test Server",
                    version="0.1.0"
                )
                logger.info("Successfully created FastMCP instance!")
                
                # Inspect instance
                instance_attrs = [attr for attr in dir(instance) if not attr.startswith('_')]
                logger.info(f"Instance attributes: {', '.join(instance_attrs)}")
                
                # Check for common methods
                for method in ['start', 'stop', 'tool']:
                    logger.info(f"Has '{method}' method: {hasattr(instance, method)}")
                
                return instance
                
            except Exception as e:
                logger.error(f"Failed to create FastMCP instance: {e}", exc_info=True)
        
        return None
        
    except ImportError as e:
        logger.error(f"Failed to import {module_name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error inspecting {module_name}: {e}", exc_info=True)
        return None

def main():
    """Main function to inspect FastMCP."""
    logger.info("=== Starting FastMCP Inspection ===")
    
    # Check Python and system info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    logger.info(f"Working directory: {Path.cwd()}")
    
    # Inspect FastMCP module
    fastmcp = inspect_module('fastmcp')
    
    if fastmcp is None:
        logger.error("Failed to inspect FastMCP module")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
