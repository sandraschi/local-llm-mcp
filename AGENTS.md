# llm-mcp — Agent Guide

## Overview
ðŸ”¥ Production-ready FastMCP 3.1.0 server for managing local and cloud LLMs with vLLM 1.0 integration and Claude Desktop compatibility

## Entry Points
- `uv run llm-mcp` → `llm_mcp.main:cli`

## Standards
- FastMCP 3.2+ portmanteau tool pattern — tools use `operation` enum param
- Responses: structured dicts with `success`, `message`, domain-specific fields
- Dual transport: stdio (Claude Desktop) + HTTP (`MCP_TRANSPORT=http`)
- See [mcp-central-docs](https://github.com/sandraschi/mcp-central-docs) for fleet-wide coding standards

## Key Files
- `README.md` — full documentation
- `pyproject.toml` — build config and entry points
- `CLAUDE.md` — Claude Code context (if present)
