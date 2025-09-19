"""Check FastMCP import and list its contents."""
import sys
import logging
import inspect

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("check_fastmcp_import")

def main():
    """Check FastMCP import and list its contents."""
    try:
        # Try to import FastMCP
        logger.info("Attempting to import fastmcp...")
        import fastmcp
        
        # Print module info
        logger.info("Successfully imported fastmcp!")
        logger.info(f"Module location: {fastmcp.__file__}")
        
        # List available attributes
        attrs = [attr for attr in dir(fastmcp) if not attr.startswith('_')]
        logger.info(f"Available attributes: {', '.join(attrs)}")
        
        # Try to find FastMCP class
        fastmcp_class = None
        for attr in attrs:
            try:
                obj = getattr(fastmcp, attr)
                if inspect.isclass(obj) and attr == 'FastMCP':
                    fastmcp_class = obj
                    break
            except:
                continue
        
        if fastmcp_class:
            logger.info("Found FastMCP class!")
            # List methods of FastMCP class
            methods = [m for m in dir(fastmcp_class) if not m.startswith('_')]
            logger.info(f"Available methods: {', '.join(methods)}")
            
            # Try to create an instance
            try:
                logger.info("Attempting to create FastMCP instance...")
                mcp = fastmcp_class(
                    name="Test Server",
                    version="0.1.0"
                )
                logger.info("Successfully created FastMCP instance!")
                return 0
            except Exception as e:
                logger.error(f"Failed to create FastMCP instance: {e}", exc_info=True)
                return 1
        else:
            logger.error("Could not find FastMCP class in the module!")
            return 1
            
    except ImportError as e:
        logger.error(f"Failed to import fastmcp: {e}")
        logger.error("Please install it with: pip install fastmcp")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
