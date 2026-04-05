"""Test script to verify FastMCP installation and basic functionality."""
import sys
import importlib.metadata
from pathlib import Path

def check_installation():
    """Check if FastMCP is installed and print version information."""
    try:
        # Check FastMCP installation
        fastmcp_version = importlib.metadata.version('fastmcp')
        print(f"✅ FastMCP is installed (version: {fastmcp_version})")
        
        # Check Pydantic version
        pydantic_version = importlib.metadata.version('pydantic')
        print(f"✅ Pydantic is installed (version: {pydantic_version})")
        
        # Check if we can import FastMCP
        try:
            from fastmcp import FastMCP
            print("✅ Successfully imported FastMCP")
            return True
        except ImportError as e:
            print(f"❌ Failed to import FastMCP: {e}")
            return False
            
    except importlib.metadata.PackageNotFoundError as e:
        print(f"❌ Package not found: {e}")
        print("\nPlease install FastMCP using:")
        print("pip install fastmcp>=2.12.0")
        return False

def main():
    """Main function to test FastMCP installation."""
    print("🔍 Checking FastMCP installation...\n")
    
    # Add the src directory to the path
    src_path = str(Path(__file__).parent / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # Check installation
    if not check_installation():
        return
    
    print("\n✅ FastMCP installation looks good!")
    print("You can now run the LLM MCP server using:")
    print("python -m llm_mcp.main")

if __name__ == "__main__":
    main()
