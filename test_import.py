"""Test FastMCP import and basic functionality."""
import sys
import logging
import asyncio
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("test_import")

def main():
    """Test FastMCP import and basic functionality."""
    try:
        # Try to import FastMCP
        logger.info("Attempting to import FastMCP...")
        from fastmcp import FastMCP
        logger.info("Successfully imported FastMCP!")
        
        # Print version
        logger.info(f"FastMCP version: {FastMCP.__version__}")
        
        # Create a simple instance
        logger.info("Creating FastMCP instance...")
        mcp = FastMCP(
            name="Test Server",
            version="0.1.0"
        )
        logger.info("Successfully created FastMCP instance!")
        
        return 0
        
    except ImportError as e:
        logger.error(f"Failed to import FastMCP: {e}")
        logger.error("Please ensure FastMCP is installed with: pip install fastmcp")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
