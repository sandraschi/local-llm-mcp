# Local LLM MCP Server

> Optional fleet **LLM control plane** — unified MCP tools for local (Ollama, LM Studio) and cloud providers, plus a web dashboard. Most fleet MCPs talk to Ollama directly via `*_SAMPLING_BASE_URL`; keep this repo for centralized model ops when you need it.

<p align="center">
  <a href="https://github.com/casey/just"><img src="https://img.shields.io/badge/just-ready_to_go-7c5cfc?style=flat-square&logo=just&logoColor=white" alt="Just"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://github.com/PrefectHQ/fastmcp"><img src="https://img.shields.io/badge/FastMCP-3.2-7c5cfc?style=flat-square" alt="FastMCP"></a>
</p>

**Install:** [INSTALL.md](INSTALL.md) · **Purpose:** [docs/PURPOSE.md](docs/PURPOSE.md) · **Manifest:** [llms.txt](llms.txt)

## Quick start

```powershell
git clone https://github.com/sandraschi/local-llm-mcp.git
cd local-llm-mcp
just bootstrap
just serve
```

Without `just`:

```powershell
uv sync
copy .env.example .env
uv run llm-mcp
```

Dashboard (optional):

```powershell
powershell -ExecutionPolicy Bypass -File web_sota\start.ps1
```

Open `http://127.0.0.1:10832` (frontend) · API `10833`.

## What it does

| Capability | Notes |
|------------|--------|
| **MCP portmanteau tools** | `llm_health`, `llm_models`, `llm_generation`, `llm_multimodal`, `llm_finetuning`, plus provider-specific tools (`llm_ollama`, `llm_lmstudio`, …) |
| **Multi-provider** | Ollama, LM Studio, OpenAI, Anthropic, Gemini, Perplexity; vLLM/HuggingFace experimental |
| **Dashboard** | Fleet launcher, live `.env` editing, basic GPU telemetry |
| **MCP bridge** | `MCP_BRIDGE_URLS` — proxy tools from other MCP servers |

See [docs/PURPOSE.md](docs/PURPOSE.md) for how this differs from per-repo Ollama sampling.

## Honest status (2026-06)

| Area | State |
|------|--------|
| Server startup + tool registration | Works; failed tool modules are skipped |
| Ollama / LM Studio / cloud APIs | Usable when endpoints and keys are set |
| `llm_generation` / model lifecycle | Partial — some paths need fixes |
| vLLM / finetuning / MoE | Optional deps; often disabled on Windows |
| Fleet default | **Not required** — peers use `http://127.0.0.1:11434/v1` directly |

Legacy individual tools: `LLM_MCP_ENABLE_LEGACY_TOOLS=true`.

## Cursor / Claude Desktop

```json
"local-llm": {
  "command": "uv",
  "args": ["--directory", "/path/to/local-llm-mcp", "run", "llm-mcp"],
  "env": {
    "PROVIDERS__OLLAMA_BASE_URL": "http://127.0.0.1:11434",
    "PROVIDERS__LMSTUDIO_BASE_URL": "http://127.0.0.1:1234"
  }
}
```

Copy `.env.example` → `.env` for provider URLs and API keys. Never commit `.env`.

## Ports

| Service | Port |
|---------|------|
| Dashboard frontend | `10832` |
| Dashboard API | `10833` |
| Ollama (external) | `11434` |
| LM Studio (external) | `1234` |

## Development

```powershell
uv sync --extra dev
uv run pytest tests\ -q
uv run ruff check src tests
```

## License

MIT — see [LICENSE](LICENSE).
