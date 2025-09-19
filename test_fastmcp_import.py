"""Test FastMCP import and basic functionality."""
import sys
import os

def main():
    print("Testing FastMCP import...")
    
    # Print Python and environment info
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # Try to import FastMCP
    try:
        import fastmcp
        print("\nSuccessfully imported fastmcp!")
        print(f"FastMCP version: {fastmcp.__version__ if hasattr(fastmcp, '__version__') else 'No version found'}")
        print(f"FastMCP location: {os.path.dirname(fastmcp.__file__) if hasattr(fastmcp, '__file__') else 'Unknown'}")
        
        # Test basic functionality
        print("\nTesting basic FastMCP functionality...")
        try:
            from fastmcp import FastMCP
            print("Successfully imported FastMCP class")
            print("Basic FastMCP functionality appears to be working!")
        except Exception as e:
            print(f"Error testing FastMCP functionality: {e}")
            
    except ImportError as e:
        print(f"\nError importing fastmcp: {e}")
        print("\nTroubleshooting steps:")
        print("1. Make sure FastMCP is installed: pip install fastmcp>=2.12.0")
        print("2. Check your PYTHONPATH environment variable")
        print("3. Check if there are multiple Python installations that might be causing conflicts")
        print(f"4. Current sys.path: {sys.path}")

if __name__ == "__main__":
    main()
