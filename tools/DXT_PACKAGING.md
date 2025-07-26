# DXT Packaging for LLM MCP Server

This document explains how to package the LLM MCP Server as a DXT (Desktop eXtension) for use with Claude Desktop and other MCP-compatible applications.

## Overview

DXT (Desktop eXtensions) is a packaging format for MCP (Model Control Protocol) servers. It allows for easy distribution and installation of MCP servers with all their dependencies.

## Prerequisites

1. Install the DXT CLI tool:
   ```bash
   npm install -g @anthropic/dxt
   ```

2. Ensure you have Python 3.8+ installed

3. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Generating the Manifest

The `dxt_generator.py` script automates the creation of the `manifest.json` file by analyzing your MCP server code.

### Basic Usage

```bash
python tools/dxt_generator.py -m src/llm_mcp/server.py -o manifest.json
```

### With Custom Configuration

1. Edit `tools/dxt_config.json` to customize the manifest:
   - Update metadata (name, version, author, etc.)
   - Configure server settings
   - Define user-configurable options

2. Generate the manifest with custom configuration:
   ```bash
   python tools/dxt_generator.py -m src/llm_mcp/server.py -o manifest.json --overrides tools/dxt_config.json
   ```

## Package Structure

A DXT package has the following structure:

```
llm-mcp.dxt (ZIP file)
├── manifest.json      # Generated manifest file
├── server/           # Server files
│   ├── main.py       # Main server entry point
│   └── ...           # Other server files
├── requirements.txt  # Python dependencies
└── icon.png         # Optional: Extension icon (128x128 recommended)
```

## Building the DXT Package

1. First, ensure you have a valid `manifest.json` file

2. Create the package structure:
   ```bash
   mkdir -p dist/llm-mcp/server
   cp -r src/llm_mcp/* dist/llm-mcp/server/
   cp requirements.txt dist/llm-mcp/
   cp manifest.json dist/llm-mcp/
   # Copy icon if you have one
   # cp assets/icon.png dist/llm-mcp/
   ```

3. Create the DXT package:
   ```bash
   cd dist
   dxt pack llm-mcp -o llm-mcp.dxt
   ```

## Installing the DXT Package

1. In Claude Desktop, go to Settings > Extensions
2. Click "Install Extension" and select the `llm-mcp.dxt` file
3. Configure any required settings
4. Restart Claude Desktop if prompted

## Development Workflow

1. Make changes to your MCP server code
2. Update tests as needed
3. Generate an updated manifest:
   ```bash
   python tools/dxt_generator.py -m src/llm_mcp/server.py -o manifest.json
   ```
4. Build and test the DXT package
5. Commit and push your changes

## Best Practices

1. **Tool Documentation**:
   - Write clear docstrings for all tools
   - Include parameter descriptions and return value information
   - Document any side effects or requirements

2. **Configuration**:
   - Make all configurable options available in `user_config`
   - Provide sensible defaults
   - Use appropriate validation for user inputs

3. **Error Handling**:
   - Include meaningful error messages
   - Handle missing dependencies gracefully
   - Validate inputs before processing

4. **Performance**:
   - Keep tool functions efficient
   - Use async/await for I/O-bound operations
   - Consider adding timeouts for long-running operations

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   - Ensure all dependencies are listed in `requirements.txt`
   - Check that the Python version matches the compatibility requirements

2. **Manifest Validation Errors**:
   - Verify that all required fields are present
   - Check for JSON syntax errors
   - Ensure tool names are unique and follow naming conventions

3. **Permission Issues**:
   - Make sure the server has permission to access required files and directories
   - Check file permissions on Unix-like systems

4. **Connection Problems**:
   - Verify that the server is binding to the correct address and port
   - Check firewall settings if running on a remote machine

## Resources

- [DXT Documentation](https://github.com/anthropics/dxt)
- [MCP Protocol Specification](https://github.com/anthropics/mcp)
- [Example MCP Servers](https://github.com/anthropics/dxt/tree/main/examples)
