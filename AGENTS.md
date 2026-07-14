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

## LM Link Integration (Tailscale + LM Studio)

The `llm_lmstudio` tool has a `link_status` operation that probes LM Link via
`lms link status --json` (async subprocess). Returns live peer list, loaded models,
link state, and preferred device.

**Provider health** (`provider_health.py`): `check_lm_link_status()` runs the same
probe with 60-second cache TTL. Results appear in:
- `GET /api/v1/health` → `lm_link` key (peers, device_name, peer_count)
- `GET /api/v1/diagnostics` → `lm_link` key (forced refresh, full peer data)

**Cross-repo**: tailscale-mcp (port 10821) is the primary LM Link control plane
(enable/disable, device naming, preferred device). local-llm-mcp reads LM Link
status for dashboard and provider awareness.

## Key Files
- `README.md` — full documentation
- `pyproject.toml` — build config and entry points
- `CLAUDE.md` — Claude Code context (if present)
- `src/llm_mcp/gateway/` — provider adapter implementations
- `src/llm_mcp/services/provider_health.py` — provider health + LM Link probes
- `src/llm_mcp/tools/portmanteau_lmstudio.py` — LM Studio + LM Link MCP tool

Install docs: follow mcp-central-docs/standards/AGENT_INSTALL_REFERENCE.md
