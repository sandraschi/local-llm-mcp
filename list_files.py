""
Simple script to list files in the src directory and check Python package structure.
"""
import os
from pathlib import Path

def list_files(directory: str, indent: int = 0):
    """Recursively list files in a directory with indentation."""
    prefix = '  ' * indent
    path = Path(directory)
    
    print(f"{prefix}{path.name}/")
    
    for item in path.iterdir():
        if item.is_dir():
            list_files(item, indent + 1)
        else:
            print(f"{prefix}  {item.name}")

if __name__ == "__main__":
    # List files in the src directory
    src_dir = Path(__file__).parent / "src"
    if src_dir.exists():
        print(f"Contents of {src_dir}:")
        list_files(src_dir)
    else:
        print(f"Directory not found: {src_dir}")
    
    # Check for __init__.py files
    print("\nChecking for __init__.py files:")
    llm_mcp_dir = src_dir / "llm_mcp"
    if llm_mcp_dir.exists():
        init_file = llm_mcp_dir / "__init__.py"
        print(f"- {init_file}: {'exists' if init_file.exists() else 'MISSING'}")
        
        tools_dir = llm_mcp_dir / "tools"
        if tools_dir.exists():
            tools_init = tools_dir / "__init__.py"
            print(f"- {tools_init}: {'exists' if tools_init.exists() else 'MISSING'}")
    else:
        print(f"LLM MCP directory not found: {llm_mcp_dir}")
