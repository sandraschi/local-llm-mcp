"""Debug FastMCP installation and import."""
import sys
import logging
import importlib
import pkg_resources
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("debug_fastmcp")

def check_package_installation(package_name):
    """Check if a package is installed and return its version."""
    try:
        dist = pkg_resources.get_distribution(package_name)
        logger.info(f"Found {package_name} version: {dist.version}")
        logger.info(f"Location: {dist.location}")
        return True, dist.version
    except pkg_resources.DistributionNotFound:
        logger.error(f"{package_name} is not installed")
        return False, None
    except Exception as e:
        logger.error(f"Error checking {package_name}: {e}")
        return False, None

def check_module_import(module_name):
    """Check if a module can be imported and list its attributes."""
    try:
        module = importlib.import_module(module_name)
        logger.info(f"Successfully imported {module_name}")
        
        # List available attributes
        attrs = [attr for attr in dir(module) if not attr.startswith('_')]
        logger.info(f"Available attributes in {module_name}: {', '.join(attrs[:20])}" + 
                   ("..." if len(attrs) > 20 else ""))
        
        # Try to get version
        version = getattr(module, '__version__', 'Not available')
        logger.info(f"{module_name} version: {version}")
        
        return True
    except ImportError as e:
        logger.error(f"Failed to import {module_name}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error importing {module_name}: {e}")
        return False

def main():
    """Main function to debug FastMCP installation."""
    logger.info("=== FastMCP Debugging Tool ===")
    
    # Check Python version
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Python path: {sys.path}")
    
    # Check FastMCP installation
    logger.info("\n[1/2] Checking FastMCP installation...")
    fastmcp_installed, fastmcp_version = check_package_installation("fastmcp")
    
    # Check FastMCP import
    logger.info("\n[2/2] Checking FastMCP import...")
    fastmcp_imported = check_module_import("fastmcp")
    
    # Summary
    logger.info("\n=== Debug Summary ===")
    logger.info(f"FastMCP installed: {'Yes' if fastmcp_installed else 'No'}")
    if fastmcp_installed:
        logger.info(f"FastMCP version: {fastmcp_version}")
    logger.info(f"FastMCP imported: {'Yes' if fastmcp_imported else 'No'}")
    
    if not fastmcp_imported:
        logger.error("\nTroubleshooting steps:")
        logger.error("1. Make sure FastMCP is installed: pip install fastmcp")
        logger.error("2. Check your Python environment matches the one you're using")
        logger.error("3. Try reinstalling: pip uninstall fastmcp && pip install --no-cache-dir fastmcp")
        logger.error("4. Check for any error messages above")
    
    return 0 if fastmcp_imported else 1

if __name__ == "__main__":
    sys.exit(main())
