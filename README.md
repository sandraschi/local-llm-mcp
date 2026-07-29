# Local LLM MCP Server

A production-grade FastMCP server for managing local and cloud LLMs: Ollama, vLLM, LM Studio, and 28 cloud providers through a unified AI gateway. Includes a SOTA React/Vite dashboard, provider health monitoring with circuit breakers, and LM Link peer discovery over Tailscale.

```
Ports:    10832 (dashboard frontend)  10833 (dashboard API + gateway)
Stack:    FastMCP 3.4.4+  |  FastAPI  |  React 19 + Vite 6 + TailwindCSS
```

## Quick Start

```powershell
just bootstrap    # install deps + pre-commit hooks
just serve        # start MCP server (stdio mode)
```

## Tool Surface (12 Portmanteau Tools)

The server consolidates 30+ operations into 12 portmanteau tools:

| Tool | Operations | Category |
|------|-----------|----------|
| `llm_health` | `health_check`, `provider_check`, `shutdown`, `system_info`, `service_status`, `get_metrics`, `set_log_level`, `collect_metrics`, `list_tools`, `tool_help`, `search_tools` | System |
| `llm_models` | `list_models`, `get_model_info`, `register_model`, `unregister_model`, `update_model`, `get_model_stats` | Models |
| `llm_generation` | `generate_text`, `chat_completion`, `embed_text`, `stream_generate` | Generation |
| `llm_multimodal` | `analyze_image`, `generate_image`, `compare_images` | Vision |
| `llm_finetuning` | `lora_load_adapter`, `lora_unload_adapter`, `lora_list_adapters` | Training |
| `llm_ollama` | `list_models`, `pull_model`, `delete_model`, `load_model`, `unload_model` | Ollama |
| `llm_lmstudio` | `list_models`, `load_model`, `unload_model`, `eject_model`, `link_status` | LM Studio |
| `llm_vllm` | `list_models`, `get_server_status`, `start_server`, `stop_server`, `get_config`, `update_config` | vLLM |
| `llm_huggingface` | `list_models`, `search_models`, `download_model`, `get_model_details`, `list_datasets` | HuggingFace |
| `llm_google_cloud` | `generate_text`, `list_models`, `generate_content`, `embed_text` | Cloud |
| `llm_gpu` | `get_status`, `clear_memory`, `optimize`, `get_health` | GPU |
| `llm_help` | `list_tools`, `get_tool_help`, `search_tools`, `get_tool_signature` | Help |

All tools return `{success, message, data}` per the fleet dialogic return standard.

## AI Gateway (Lightport-compatible)

`POST /v1/chat/completions` — OpenAI-compatible proxy to 28 providers.

Select provider by header (`x-lightport-provider`) or model prefix (`anthropic/`):

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:10833/v1", api_key="sk-...")
client.default_headers["x-lightport-provider"] = "deepseek"
resp = client.chat.completions.create(model="deepseek-chat", messages=[...])
```

### Local Providers
| Provider | Port | Auto-detected |
|----------|------|---------------|
| Ollama | 11434 | Health check with circuit breaker |
| LM Studio | 1234 | Health check + LM Link probe |
| vLLM | 8000 | Docker-managed lifecycle |

### Cloud Providers (28)
Anthropic, Azure, Bedrock, Cohere, DeepInfra, DeepSeek, Featherless, Fireworks, Gemini, Groq, Hyperbolic, Lepton, Mistral, Modal, Nebius, Novita, OpenAI, OpenRouter, Perplexity, Replicate, SambaNova, SiliconFlow, Together, xAI (Grok), Anyscale

## Provider Health Service

Built-in health monitoring with:
- **Liveness checks**: Probes each provider every 60s (cached)
- **Circuit breaker**: 3 consecutive failures → 60s cooldown before retry
- **Structured results**: `ProviderHealth` dataclass with `ok`, `reachable`, `model_count`, `error`, `latency_ms`
- **LM Link probe**: Runs `lms link status --json` for Tailscale peer discovery
- **Endpoints**: `GET /api/v1/health`, `GET /api/v1/diagnostics`, `GET /v1/gateway/providers/health`

## Web Dashboard

React 19 + Vite 6 app at `http://localhost:10832`:

| Page | Route | Features |
|------|-------|----------|
| Dashboard | `/` | KPI cards, provider health, LM Link status, backend connection dot |
| Chat | `/chat` | Skill-aware chat, 4+ personalities, export/clear, speech TTS/STT |
| Performance | `/performance` | GPU VRAM, system RAM, latency telemetry |
| Vision | `/vision` | Multimodal model inference |
| Fleet | `/fleet` | MCP ecosystem app discovery |
| Analytics | `/analytics` | Usage statistics and metrics |
| Settings | `/settings` | Provider config, LLM detection, server settings |
| Help | `/help` | Architecture, ports, env vars, troubleshooting |

## LM Link Integration (Tailscale + LM Studio)

The `llm_lmstudio(operation="link_status")` operation probes LM Link — Tailscale-powered mesh for remote LLM access. Returns live peer list, loaded models, link state, and preferred device. Provider health endpoints also include LM Link data under `lm_link`.

Cross-repo: LM Link network control lives in [tailscale-mcp](https://github.com/sandraschi/tailscale-mcp).

## Architecture

```
MCP Client ──→ FastMCP (stdio/HTTP) ──→ Portmanteau Tools ──→ Provider Layer
                                       (12 tools)               ├── Ollama
                                      ┌─ Health (system)        ├── LM Studio
                                      ├─ Models (discovery)     ├── vLLM
                                      ├─ Generation             ├── HuggingFace
                                      ├─ Multimodal             ├── Anthropic
                                      ├─ Finetuning             ├── Gemini
                                      ├─ Ollama-specific        ├── OpenAI
                                      ├─ LM Studio + LM Link    └── ...22 more
                                      ├─ vLLM
                                      ├─ HuggingFace
                                      ├─ Google Cloud
                                      ├─ GPU
                                      └─ Help

Web Dashboard ──→ FastAPI (10833) ──→ Health/Config/Diagnostics
                                 └── Gateway (/v1/chat/completions)
```

## Configuration

Environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_TRANSPORT` | `stdio` | Transport mode: stdio, http, sse |
| `MCP_PORT` | `10833` | HTTP mode port |
| `MCP_HOST` | `127.0.0.1` | HTTP mode bind address |
| `LLM_MCP_LOG_LEVEL` | `WARNING` | Log level: DEBUG, INFO, WARNING, ERROR |
| `LLM_MCP_ENABLE_LEGACY_TOOLS` | `false` | Register individual (non-portmanteau) tools |

## Session Context Injection

The server injects a tool-awareness prompt at IDE session start via:
- `.claude-plugin/plugin.json` + `hooks/hooks.json` (Claude Code)
- `.cursorrules` (Cursor)
- `.windsurfrules` (Windsurf)
- `.github/copilot-instructions.md` (GitHub Copilot)
- `.opencode/skills/session-context/SKILL.md` (OpenCode)

## Development

```powershell
just bootstrap    # uv sync + pre-commit + npm ci
just lint         # ruff check + biome ci
just fix          # ruff --fix + ruff format + biome check --write
just test         # pytest tests/
just serve        # uv run -m llm_mcp
just certify      # lint + test
just mcpb-pack    # build MCPB bundle
```

## License

MIT
