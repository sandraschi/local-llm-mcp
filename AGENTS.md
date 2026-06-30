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

## AI Gateway (Lightport-compatible)

`POST /v1/chat/completions` — OpenAI-compatible proxy routing to 28 providers.
Select provider via `x-lightport-provider` header or model prefix (e.g. `anthropic/...`).

| Local | Cloud |
|-------|-------|
| Ollama, LM Studio, vLLM | Anthropic, Azure, Bedrock, Cohere, DeepInfra, DeepSeek, Featherless, Fireworks, Gemini, Groq, Hyperbolic, Lepton, Mistral, Modal, Nebius, Novita, OpenAI, OpenRouter, Perplexity, Replicate, SambaNova, SiliconFlow, Together, xAI (Grok), Anyscale |

Usage: `client = OpenAI(base_url="http://127.0.0.1:10833/v1", api_key="...")` then set `client.default_headers["x-lightport-provider"] = "anthropic"`. See `src/llm_mcp/gateway/README.md`.

## Key Files
- `README.md` — full documentation
- `pyproject.toml` — build config and entry points
- `CLAUDE.md` — Claude Code context (if present)
- `src/llm_mcp/gateway/` — provider adapter implementations

Install docs: follow mcp-central-docs/standards/AGENT_INSTALL_REFERENCE.md
