# DXT Packaging for LLM MCP Server

This document explains how to package the LLM MCP Server as a DXT (Desktop eXtension) for use with Claude Desktop and
other MCP-compatible applications.

## Overview

DXT (Desktop eXtensions) is a packaging format for MCP (Model Control Protocol) servers. It allows for easy distribution
and installation of MCP servers with all their dependencies.

## Prerequisites

$11. Install the DXT CLI tool:

   ```bash
   npm install -g @anthropic/dxt
   ```

$11. Ensure you have Python 3.8+ installed

$11. Install project dependencies:

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

$11. Edit `tools/dxt_config.json` to customize the manifest:

   - Update metadata (name, version, author, etc.)


   - Configure server settings

   - Define user-configurable options



$11. Generate the manifest with custom configuration:

   ```bash
   python tools/dxt_generator.py -m src/llm_mcp/server.py -o manifest.json --overrides tools/dxt_config.json
   ```

## Package Structure

A DXT package has the following structure:

```
llm-mcp.dxt (ZIP file)





â”œâ”€â”€ manifest.json      # Generated manifest file
â”œâ”€â”€ server/           # Server files
â”‚   â”œâ”€â”€ main.py       # Main server entry point
â”‚   â””â”€â”€ ...           # Other server files
â”œâ”€â”€ requirements.txt  # Python dependencies
â””â”€â”€ icon.png         # Optional: Extension icon (128x128 recommended)

```


## Building the DXT Package

$11. First, ensure you have a valid `manifest.json` file

$11. Create the package structure:

   ```bash
   mkdir -p dist/llm-mcp/server
   cp -r src/llm_mcp/* dist/llm-mcp/server/
   cp requirements.txt dist/llm-mcp/
   cp manifest.json dist/llm-mcp/
   # Copy icon if you have one
   # cp assets/icon.png dist/llm-mcp/
   ```

$11. Create the DXT package:

   ```bash
   cd dist
   dxt pack llm-mcp -o llm-mcp.dxt
   ```

## Installing the DXT Package

$11. In Claude Desktop, go to Settings > Extensions

$11. Click "Install Extension" and select the `llm-mcp.dxt` file


$11. Configure any required settings

$11. Restart Claude Desktop if prompted



## Development Workflow

$11. Make changes to your MCP server code

$11. Update tests as needed


$11. Generate an updated manifest:

   ```bash
   python tools/dxt_generator.py -m src/llm_mcp/server.py -o manifest.json
   ```

$11. Build and test the DXT package

$11. Commit and push your changes



## Best Practices

$11. **Tool Documentation**:

   - Write clear docstrings for all tools


   - Include parameter descriptions and return value information

   - Document any side effects or requirements



$11. **Configuration**:

   - Make all configurable options available in `user_config`


   - Provide sensible defaults

   - Use appropriate validation for user inputs



$11. **Error Handling**:

   - Include meaningful error messages


   - Handle missing dependencies gracefully

   - Validate inputs before processing



$11. **Performance**:

   - Keep tool functions efficient


   - Use async/await for I/O-bound operations

   - Consider adding timeouts for long-running operations



## Troubleshooting

### Common Issues

$11. **Missing Dependencies**:

   - Ensure all dependencies are listed in `requirements.txt`


   - Check that the Python version matches the compatibility requirements

$11. **Manifest Validation Errors**:

   - Verify that all required fields are present


   - Check for JSON syntax errors

   - Ensure tool names are unique and follow naming conventions



$11. **Permission Issues**:

   - Make sure the server has permission to access required files and directories


   - Check file permissions on Unix-like systems

$11. **Connection Problems**:

   - Verify that the server is binding to the correct address and port


   - Check firewall settings if running on a remote machine

## Resources

- [DXT Documentation](https://github.com/anthropics/dxt)

- [MCP Protocol Specification](https://github.com/anthropics/mcp)


- [Example MCP Servers](https://github.com/anthropics/dxt/tree/main/examples)
