"""Test script for the minimal FastMCP server."""
import asyncio
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("test_minimal")

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_fastmcp_import():
    """Test if FastMCP can be imported."""
    try:
        from fastmcp import FastMCP
        logger.info("✓ FastMCP imported successfully")
        return True
    except ImportError as e:
        logger.error(f"✗ Failed to import FastMCP: {e}")
        return False

def test_minimal_server():
    """Test the minimal server."""
    try:
        from src.llm_mcp.minimal_server import MinimalServer
        logger.info("✓ Minimal server imported successfully")
        return True
    except ImportError as e:
        logger.error(f"✗ Failed to import minimal server: {e}")
        return False

async def run_tests():
    """Run all tests."""
    logger.info("Starting tests...")
    
    # Test 1: Check FastMCP import
    logger.info("\n--- Testing FastMCP import ---")
    import_ok = test_fastmcp_import()
    
    # Test 2: Check minimal server import
    logger.info("\n--- Testing minimal server import ---")
    server_import_ok = test_minimal_server()
    
    # Print summary
    logger.info("\n=== Test Summary ===")
    logger.info(f"FastMCP import: {'✓' if import_ok else '✗'}")
    logger.info(f"Minimal server import: {'✓' if server_import_ok else '✗'}")
    
    if not (import_ok and server_import_ok):
        logger.error("\nSome tests failed. Please check the logs above for details.")
        return 1
    
    logger.info("\nAll tests passed! You can now run the minimal server with:")
    logger.info("python -m src.llm_mcp.minimal_server")
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(run_tests())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
