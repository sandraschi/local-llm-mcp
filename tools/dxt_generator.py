#!/usr/bin/env python3
"""
DXT Manifest Generator for LLM MCP Server

This script automatically generates a manifest.json file for DXT packaging by analyzing
the MCP server code and extracting tool definitions, parameters, and documentation.
"""

import os
import re
import json
import inspect
import argparse
import importlib
import sys
import types
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Union, get_type_hints

# Default values for the manifest
DEFAULT_MANIFEST = {
    "$schema": "https://raw.githubusercontent.com/anthropics/dxt/main/dist/dxt-manifest.schema.json",
    "dxt_version": "0.1",
    "name": "llm-mcp-server",
    "display_name": "LLM MCP Server",
    "version": "0.1.0",
    "description": "Multi-provider LLM MCP server with support for local and cloud models",
    "long_description": "A Model Control Protocol (MCP) server that provides a unified interface to multiple LLM providers including Ollama, Anthropic, and others. Supports model listing, text generation, and conversation management.",
    "author": {
        "name": "Your Name",
        "email": "your.email@example.com",
        "url": "https://github.com/yourusername/llm-mcp"
    },
    "repository": {
        "type": "git",
        "url": "https://github.com/yourusername/llm-mcp"
    },
    "homepage": "https://github.com/yourusername/llm-mcp",
    "documentation": "https://github.com/yourusername/llm-mcp#readme",
    "support": "https://github.com/yourusername/llm-mcp/issues",
    "icon": "icon.png",
    "server": {
        "type": "python",
        "entry_point": "server/main.py",
        "mcp_config": {
            "command": "python",
            "args": [
                "${__dirname}/server/main.py"
            ],
            "env": {
                "PYTHONPATH": "${__dirname}/server"
            }
        }
    },
    "tools": [],
    "keywords": ["llm", "mcp", "ai", "ollama", "anthropic", "local-ai"],
    "license": "MIT",
    "user_config": {},
    "compatibility": {
        "claude_desktop": ">=0.10.0",
        "platforms": ["darwin", "win32", "linux"],
        "runtimes": {
            "python": ">=3.8.0 <4"
        }
    }
}

# Type to schema mapping for user config
type_to_schema = {
    'str': {'type': 'string'},
    'int': {'type': 'number'},
    'float': {'type': 'number'},
    'bool': {'type': 'boolean'},
    'path': {'type': 'string', 'format': 'path'},
    'file': {'type': 'string', 'format': 'file'},
    'directory': {'type': 'string', 'format': 'directory'}
}

class ToolParameter:
    """Represents a parameter for an MCP tool."""
    
    def __init__(self, name: str, type_: Type, default: Any = None, 
                 description: str = None, required: bool = True):
        self.name = name
        self.type = type_
        self.default = default
        self.description = description or ""
        self.required = required and (default is None)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameter to dictionary for JSON serialization."""
        type_name = self.type.__name__.lower()
        schema = type_to_schema.get(type_name, {"type": "string"}).copy()
        
        if self.default is not None:
            schema["default"] = self.default
        
        if self.description:
            schema["description"] = self.description
        
        schema["title"] = self.name.replace('_', ' ').title()
        
        return schema


class ToolDefinition:
    """Represents an MCP tool with its metadata and parameters."""
    
    def __init__(self, name: str, func: callable):
        self.name = name
        self.func = func
        self.docstring = inspect.getdoc(func) or ""
        self.parameters = self._extract_parameters()
    
    def _extract_parameters(self) -> Dict[str, ToolParameter]:
        """Extract parameter information from the function signature."""
        sig = inspect.signature(self.func)
        type_hints = get_type_hints(self.func, include_extras=True)
        
        parameters = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            param_type = type_hints.get(param_name, str)
            
            # Handle Python 3.10+ union types (using | operator)
            if hasattr(param_type, '__or__') and hasattr(param_type, '__args__'):
                # Convert to typing.Union for consistent handling
                param_type = Union[param_type.__args__]
            
            # Handle Optional[] and Union types
            if hasattr(param_type, '__origin__') and param_type.__origin__ is not None:
                # Handle Optional[Type] which is Union[Type, None]
                if (param_type.__origin__ is Union and 
                    type(None) in param_type.__args__ and 
                    len(param_type.__args__) == 2):
                    # Get the non-None type from Optional[Type] -> Union[Type, None]
                    param_type = next(t for t in param_type.__args__ if t is not type(None))
                # Handle other Union types
                elif param_type.__origin__ is Union:
                    # For now, just take the first non-None type or default to str
                    non_none_types = [t for t in param_type.__args__ if t is not type(None)]
                    param_type = non_none_types[0] if non_none_types else str
            
            # Handle other complex types (List, Dict, etc.) by converting to string for now
            if hasattr(param_type, '__origin__') and param_type.__origin__ not in (Union, type(None)):
                param_type = str
                
            # Handle types from the types module (like types.UnionType in Python 3.10+)
            if hasattr(param_type, '__module__') and param_type.__module__ == 'types':
                param_type = str
            
            # Extract description from docstring
            param_desc = self._extract_param_description(param_name)
            
            parameters[param_name] = ToolParameter(
                name=param_name,
                type_=param_type,
                default=param.default if param.default != inspect.Parameter.empty else None,
                description=param_desc,
                required=(param.default == inspect.Parameter.empty)
            )
        
        return parameters
    
    def _extract_param_description(self, param_name: str) -> str:
        """Extract parameter description from docstring."""
        if not self.docstring:
            return ""
            
        # Look for :param param_name: description pattern
        pattern = rf':param\s+{param_name}:(.*?)(?=\n\s*:param|\s*:return:|\s*:raises:|\s*$)'
        match = re.search(pattern, self.docstring, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        return ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool definition to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.docstring.split('\n')[0] if self.docstring else "",
            "parameters": {
                "type": "object",
                "properties": {
                    name: param.to_dict()
                    for name, param in self.parameters.items()
                },
                "required": [
                    name for name, param in self.parameters.items()
                    if param.required
                ]
            }
        }


def get_module_name_from_path(module_path: str, project_root: str) -> str:
    """Convert a file path to a module path relative to the project root."""
    try:
        # Try to get relative path first
        rel_path = Path(module_path).relative_to(project_root)
    except ValueError:
        # If not a subpath, use the module path as is
        rel_path = Path(module_path)
    
    # Convert path to module name
    module_parts = []
    for part in rel_path.parts:
        if part.endswith('.py'):
            module_parts.append(part[:-3])  # Remove .py
        elif part != '..' and part != '.':
            module_parts.append(part)
    
    return '.'.join(module_parts)

def setup_import_paths(module_path: str) -> tuple:
    """Set up Python import paths and return the module name and directory."""
    # Get absolute path to the module
    module_path = os.path.abspath(module_path)
    module_dir = os.path.dirname(module_path)
    
    # Add the parent directory to Python path to handle package imports
    parent_dir = os.path.dirname(module_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Determine the package name (llm_mcp)
    package_name = os.path.basename(module_dir)
    
    return module_path, module_dir, package_name

def discover_tools(module_path: str) -> Dict[str, ToolDefinition]:
    """Discover MCP tools in a Python module and its submodules."""
    tools = {}
    
    try:
        print(f"Discovering tools in module: {module_path}")
        
        # Set up import paths and get module info
        module_path, module_dir, package_name = setup_import_paths(module_path)
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        
        # Create a proper module spec with the full package path
        full_module_name = f"{package_name}.{module_name}" if module_name != "__init__" else package_name
        
        print(f"Importing module: {full_module_name} from {module_path}")
        
        # Import the module with proper package context
        spec = importlib.util.spec_from_file_location(
            full_module_name, 
            module_path,
            submodule_search_locations=[module_dir]
        )
        
        if spec is None or spec.loader is None:
            print(f"Error: Could not create spec for module {module_path}")
            return tools
        
        # Create and execute the module with proper package context
        module = importlib.util.module_from_spec(spec)
        module.__package__ = package_name
        sys.modules[full_module_name] = module
        
        # Store the original FastMCP import if it exists
        original_fastmcp = sys.modules.get('fastmcp')
        
        try:
            # Execute the module
            spec.loader.exec_module(module)
            
            # Try to get the FastMCP instance from the module
            mcp_instance = None
            for name, obj in inspect.getmembers(module):
                if isinstance(obj, type) and 'FastMCP' in str(obj):
                    print(f"Found FastMCP class: {obj}")
                elif 'FastMCP' in str(type(getattr(obj, '__class__', ''))):
                    print(f"Found FastMCP instance: {obj}")
                    mcp_instance = obj
            
            # If we found an MCP instance, try to get tools from it
            if mcp_instance and hasattr(mcp_instance, '_tool_manager'):
                print("Found MCP instance with tool manager, attempting to get tools...")
                try:
                    # Try to get tools using the FastMCP 2.10.2 API
                    import asyncio
                    
                    # Create a new event loop for this thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # Get tools from the MCP instance
                    mcp_tools = loop.run_until_complete(mcp_instance.get_tools())
                    print(f"Found {len(mcp_tools)} tools from MCP instance")
                    
                    # Convert MCP tools to our ToolDefinition format
                    for tool_name, tool in mcp_tools.items():
                        print(f"Processing tool: {tool_name}")
                        # Create a wrapper function that matches the expected signature
                        async def tool_wrapper(**kwargs):
                            return await tool.afunc(**kwargs)
                        
                        # Copy attributes from the original tool
                        tool_wrapper.__name__ = tool_name
                        tool_wrapper.__doc__ = tool.description or f"Tool: {tool_name}"
                        
                        # Add to our tools dictionary
                        tools[tool_name] = ToolDefinition(tool_name, tool_wrapper)
                    
                    # Close the event loop
                    loop.close()
                    
                except Exception as e:
                    print(f"Error getting tools from MCP instance: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Fall back to scanning for decorated functions and setup_mcp
            if not tools:
                print("No tools found via MCP instance, falling back to scanning for decorated functions...")
                
                # First check the main module for directly decorated tools
                for name, obj in inspect.getmembers(module):
                    if not inspect.isfunction(obj):
                        continue
                        
                    # Check for @mcp.tool() decorator (supports multiple possible attribute names)
                    tool_attrs = [
                        '_mcp_tool_info',  # Standard FastMCP
                        '_tool_info',      # Alternative FastMCP
                        'mcp_tool',        # Another possible attribute
                        'is_mcp_tool'      # Yet another possibility
                    ]
                    
                    if any(hasattr(obj, attr) for attr in tool_attrs):
                        tool_name = getattr(obj, "_mcp_tool_name", name)
                        print(f"Found tool via decorator: {tool_name}")
                        tools[tool_name] = ToolDefinition(tool_name, obj)
                
                # If still no tools found, look for a setup_mcp function
                if not tools and hasattr(module, 'setup_mcp'):
                    print("Found setup_mcp function, extracting tools from it...")
                    
                    # Create a dummy MCP instance to capture tool registrations
                    class DummyMCP:
                        def __init__(self):
                            self.tools = {}
                            self.app = type('DummyApp', (), {'state': type('DummyState', (), {})})()
                        
                        def tool(self, func=None, **kwargs):
                            def decorator(f):
                                tool_name = kwargs.get('name', f.__name__)
                                self.tools[tool_name] = f
                                # Add the tool info to the function for later extraction
                                f._mcp_tool_info = True
                                f._mcp_tool_name = tool_name
                                return f
                            return decorator if func is None else decorator(func)
                    
                    dummy_mcp = DummyMCP()
                    
                    try:
                        # Call setup_mcp with our dummy MCP to capture tool registrations
                        module.setup_mcp(dummy_mcp)
                        
                        # Add the discovered tools to our tools dictionary
                        for tool_name, tool_func in dummy_mcp.tools.items():
                            print(f"Found tool in setup_mcp: {tool_name}")
                            tools[tool_name] = ToolDefinition(tool_name, tool_func)
                    except Exception as e:
                        print(f"Warning: Error extracting tools from setup_mcp: {e}")
                        import traceback
                        traceback.print_exc()
            
            # If we still don't have tools, try to find them in submodules
            if not tools:
                print("No tools found in main module, searching submodules...")
                # Find the package root (directory with __init__.py)
                package_dir = os.path.dirname(module_path)
                
                # Walk through all Python files in the same directory
                for root, _, files in os.walk(package_dir):
                    for file in files:
                        if not file.endswith('.py') or file == '__init__.py':
                            continue
                            
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, os.path.dirname(module_path))
                        
                        # Skip if this is the same file we already processed
                        if os.path.normpath(file_path) == os.path.normpath(module_path):
                            continue
                        
                        # Calculate the module name
                        rel_module = os.path.splitext(rel_path)[0].replace(os.sep, '.')
                        submodule_name = f"{module_name}.{rel_module}" if rel_module else module_name
                        
                        try:
                            # Skip files that might cause issues (like other main.py files)
                            if os.path.basename(file_path) == 'main.py' and not file_path.endswith('llm_mcp/main.py'):
                                print(f"Skipping potential conflicting main.py: {file_path}")
                                continue
                                
                            print(f"Importing submodule: {submodule_name} from {file_path}")
                            
                            try:
                                # Import the module
                                spec = importlib.util.spec_from_file_location(submodule_name, file_path)
                                if spec is None or spec.loader is None:
                                    print(f"Warning: Could not create spec for {file_path}")
                                    continue
                                    
                                # Skip modules that have already been imported to avoid reimporting
                                if submodule_name in sys.modules:
                                    print(f"Skipping already imported module: {submodule_name}")
                                    submodule = sys.modules[submodule_name]
                                else:
                                    submodule = importlib.util.module_from_spec(spec)
                                    sys.modules[submodule_name] = submodule
                                    
                                    # Execute the module in a controlled environment
                                    try:
                                        spec.loader.exec_module(submodule)
                                    except Exception as e:
                                        print(f"Warning: Could not execute module {submodule_name}: {e}")
                                        continue
                                
                                # Find tools in the submodule
                                for name, obj in inspect.getmembers(submodule):
                                    if not inspect.isfunction(obj):
                                        continue
                                        
                                    # Check for any known tool decorator attributes
                                    if any(hasattr(obj, attr) for attr in ['_mcp_tool_info', '_tool_info', 'mcp_tool', 'is_mcp_tool']):
                                        tool_name = getattr(obj, "_mcp_tool_name", name)
                                        print(f"Found tool in submodule {submodule_name}: {tool_name}")
                                        tools[tool_name] = ToolDefinition(tool_name, obj)
                            except Exception as e:
                                print(f"Error importing {submodule_name}: {e}")
                                continue
                        except Exception as e:
                            print(f"Warning: Could not import {submodule_name}: {e}")
                            import traceback
                            traceback.print_exc()
            
            return tools
            
        finally:
            # Restore the original FastMCP module if it existed
            if original_fastmcp is not None:
                sys.modules['fastmcp'] = original_fastmcp
    
    except Exception as e:
        print(f"Error discovering tools: {e}")
        import traceback
        traceback.print_exc()
        return tools


def generate_manifest(
    output_file: str,
    module_path: str,
    manifest_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate a DXT manifest.json file."""
    # Start with default manifest
    manifest = DEFAULT_MANIFEST.copy()
    
    # Apply any overrides
    if manifest_overrides:
        manifest.update(manifest_overrides)
    
    # Discover tools in the module
    try:
        tools = discover_tools(module_path)
        manifest["tools"] = [tool.to_dict() for tool in tools.values()]
    except Exception as e:
        print(f"Warning: Could not discover tools: {e}")
    
    # Write the manifest to a file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"Generated DXT manifest: {output_file}")
    return manifest


def main():
    """Main entry point for the DXT manifest generator."""
    parser = argparse.ArgumentParser(description='Generate DXT manifest.json for an MCP server')
    parser.add_argument('--module', '-m', required=True, 
                       help='Path to the Python module containing MCP tools')
    parser.add_argument('--output', '-o', default='manifest.json',
                       help='Output file path (default: manifest.json)')
    parser.add_argument('--overrides', type=str,
                       help='JSON file with manifest overrides')
    
    args = parser.parse_args()
    
    # Load overrides if provided
    overrides = {}
    if args.overrides and os.path.isfile(args.overrides):
        try:
            with open(args.overrides, 'r', encoding='utf-8') as f:
                overrides = json.load(f)
        except Exception as e:
            print(f"Error loading overrides: {e}")
            return 1
    
    # Generate the manifest
    try:
        generate_manifest(args.output, args.module, overrides)
        return 0
    except Exception as e:
        print(f"Error generating manifest: {e}")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
