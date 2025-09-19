"""Script to check FastMCP installation and environment."""
import sys
import os
import pkg_resources
import importlib.metadata

def check_python_version():
    """Check Python version."""
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Python path: {sys.path}\n")

def check_fastmcp_installed():
    """Check if FastMCP is installed and its version."""
    try:
        # Try to get version using importlib (preferred for Python 3.8+)
        version = importlib.metadata.version('fastmcp')
        print(f"FastMCP is installed (via importlib): {version}")
        return True
    except importlib.metadata.PackageNotFoundError:
        print("FastMCP not found via importlib")
    
    try:
        # Try to get version using pkg_resources (legacy)
        version = pkg_resources.get_distribution('fastmcp').version
        print(f"FastMCP is installed (via pkg_resources): {version}")
        return True
    except pkg_resources.DistributionNotFound:
        print("FastMCP not found via pkg_resources")
    
    return False

def check_fastmcp_import():
    """Check if FastMCP can be imported."""
    try:
        import fastmcp
        print(f"Successfully imported fastmcp from: {fastmcp.__file__}")
        if hasattr(fastmcp, '__version__'):
            print(f"FastMCP version (from module): {fastmcp.__version__}")
        return True
    except ImportError as e:
        print(f"Error importing fastmcp: {e}")
        return False

def check_environment():
    """Check environment variables and paths."""
    print("\nEnvironment variables:")
    for var in ['PYTHONPATH', 'PATH']:
        print(f"{var}: {os.environ.get(var, 'Not set')}")

def main():
    """Main function to run all checks."""
    print("=== FastMCP Installation Check ===\n")
    
    # Check Python environment
    print("=== Python Environment ===")
    check_python_version()
    
    # Check if FastMCP is installed
    print("=== FastMCP Installation Check ===")
    installed = check_fastmcp_installed()
    
    # Try to import FastMCP
    print("\n=== FastMCP Import Check ===")
    if installed:
        imported = check_fastmcp_import()
    else:
        print("Skipping import check - FastMCP not installed")
    
    # Check environment
    check_environment()
    
    print("\n=== Check Complete ===")

if __name__ == "__main__":
    main()
