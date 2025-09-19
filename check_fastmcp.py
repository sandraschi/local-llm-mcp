"""Check FastMCP installation and basic functionality."""
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
logger = logging.getLogger("check_fastmcp")

def check_import():
    """Check if FastMCP can be imported."""
    try:
        import fastmcp
        logger.info(f"Successfully imported fastmcp version: {fastmcp.__version__}")
        return True
    except ImportError as e:
        logger.error(f"Failed to import fastmcp: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error importing fastmcp: {e}")
        return False

def check_installation():
    """Check FastMCP installation details."""
    import importlib.util
    import pkg_resources
    
    try:
        # Check if package is installed
        spec = importlib.util.find_spec("fastmcp")
        if spec is None:
            logger.error("fastmcp package not found in Python path")
            return False
            
        # Get package version
        try:
            version = pkg_resources.get_distribution("fastmcp").version
            logger.info(f"fastmcp version: {version}")
            
            # Check minimum version requirement
            min_version = "2.12.0"
            if pkg_resources.parse_version(version) < pkg_resources.parse_version(min_version):
                logger.error(f"fastmcp version {min_version} or higher is required")
                return False
                
        except pkg_resources.DistributionNotFound:
            logger.error("fastmcp package is not properly installed")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error checking installation: {e}")
        return False

async def test_fastmcp():
    """Test basic FastMCP functionality."""
    try:
        from fastmcp import FastMCP
        
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
            
        logger.info("FastMCP instance created successfully")
        logger.info(f"Registered tools: {[t.name for t in mcp.tools]}")
        
        # Test the tool directly
        logger.info("Testing hello tool...")
        result = await hello("FastMCP Tester")
        logger.info(f"Tool result: {result}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing FastMCP: {e}", exc_info=True)
        return False

async def main():
    """Run all checks."""
    logger.info("=== Starting FastMCP Check ===")
    
    # Check import
    logger.info("\n[1/3] Checking FastMCP import...")
    if not check_import():
        logger.error("FastMCP import check failed")
        return 1
        
    # Check installation
    logger.info("\n[2/3] Checking FastMCP installation...")
    if not check_installation():
        logger.error("FastMCP installation check failed")
        return 1
        
    # Test functionality
    logger.info("\n[3/3] Testing FastMCP functionality...")
    if not await test_fastmcp():
        logger.error("FastMCP functionality test failed")
        return 1
        
    logger.info("\n=== All checks passed successfully ===")
    return 0

if __name__ == "__main__":
    try:
        # Configure asyncio for Windows
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Check interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
