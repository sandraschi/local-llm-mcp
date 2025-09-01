"""Simple test script to verify basic imports."""
import sys
import traceback
from pathlib import Path

# Add the src directory to the path
src_path = str(Path(__file__).parent / "src")
sys.path.insert(0, src_path)
print(f"Added to Python path: {src_path}")

# Print Python path
print("\nPython path:")
for p in sys.path:
    print(f"- {p}")

# Try to import FastMCP
try:
    print("\nAttempting to import FastMCP...")
    from fastmcp import FastMCP
    print("✓ Successfully imported FastMCP"
    print(f"FastMCP version: {FastMCP.__version__ if hasattr(FastMCP, '__version__') else 'N/A'}")
except Exception as e:
    print(f"✗ Failed to import FastMCP: {e}")
    print(traceback.format_exc())

# Try to import llm_mcp
try:
    print("\nAttempting to import llm_mcp...")
    import llm_mcp
    print(f"✓ Successfully imported llm_mcp from {llm_mcp.__file__}")
    
    # List available modules
    print("\nAvailable modules in llm_mcp:")
    for attr in dir(llm_mcp):
        if not attr.startswith('_'):
            print(f"- {attr}")
    
    # Try to import tools
    try:
        from llm_mcp.tools import check_dependencies
        print("\n✓ Successfully imported check_dependencies")
        
        # Check dependencies
        print("\nChecking dependencies...")
        deps = check_dependencies()
        print("Dependency check results:")
        for name, status in deps.items():
            print(f"- {name}: {'✓' if status else '✗'}")
            
    except ImportError as e:
        print(f"\n✗ Failed to import from llm_mcp.tools: {e}")
        print(traceback.format_exc())
    
except Exception as e:
    print(f"✗ Failed to import llm_mcp: {e}")
    print(traceback.format_exc())

print("\nTest completed.")
