# Purpose: local-llm-mcp vs fleet Ollama glom-ons

## Role

`local-llm-mcp` is an **optional hub** — not the default inference path for the Sandra MCP fleet.

- **Hub use:** One MCP surface to list models, call generation, switch local/cloud providers, and edit LLM config from the dashboard (`10832`/`10833`).
- **Typical fleet use:** Each MCP server sets `*_SAMPLING_BASE_URL=http://127.0.0.1:11434/v1` and calls Ollama directly for `agentic_*` / sampling workflows.

## When to use this repo

| Scenario | Use local-llm-mcp? |
|----------|-------------------|
| Agent needs `llm_models` / `llm_generation` as explicit MCP tools | Yes |
| Central dashboard for provider URLs and API keys | Yes |
| Route multiple backends (Ollama + LM Studio + cloud) from one place | Yes |
| Simple local chat in another repo’s `web_sota` | No — use Ollama/LM Studio UI |
| Server-side sampling inside jellyfin-mcp, arxiv-mcp, etc. | No — direct Ollama URL is enough |

## What we keep maintained

- Honest README and `llms.txt` / `llms-full.txt` (no auto-dump of unrelated docs)
- Portmanteau tools that register cleanly even when heavy deps (vLLM, finetuning) are absent
- Dashboard start script aligned with [mcp-central-docs WEBAPP_PORTS](https://github.com/sandraschi/mcp-central-docs/blob/main/operations/WEBAPP_PORTS.md)

## What is out of scope for “just in case”

- Replacing Ollama across the whole fleet
- Promising vLLM “19× faster” on every machine
- Shipping notepad++-mcp template docs as if they were LLM-specific
