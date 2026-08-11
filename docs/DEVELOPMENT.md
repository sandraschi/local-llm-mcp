# Development

## Tools Required

```
winget install astral-sh.uv
winget install Git.Git
winget install Casey.Just
winget install OpenJS.NodeJS
```

Verify: `uv --version && git --version && just --version && node --version`

## Setup

```
git clone https://github.com/sandraschi/local-llm-mcp
cd local-llm-mcp
uv sync --extra dev
cd web_sota && npm ci && cd ..
```

## Common Tasks

| Command | Description |
|---------|-------------|
| `just lint` | Ruff lint (Python) + Biome lint (webapp) |
| `just fix` | Auto-fix lint issues |
| `just test` | Run pytest |
| `just serve` | Start the MCP server (stdio) |
| `just cert` | Lint + Test gate |
| `just mcpb-pack` | Build .mcpb bundle |

## Project Structure

```
src/llm_mcp/
├── main.py              # CLI entry point
├── server.py            # FastMCP server setup
├── config.py            # Configuration
├── state.py             # State management
├── transport.py         # Transport layer
├── api/                 # REST API endpoints
├── core/                # Core logic
├── gateway/             # 28-provider AI gateway
├── providers/           # Provider implementations
├── services/            # Business logic (provider health, etc.)
├── tools/               # Portmanteau MCP tools
└── utils/               # Logging, GPU utilities

web_sota/                # React + Vite + Tailwind dashboard
docs/                    # Documentation
```

## Code Standards

See `mcp-central-docs/standards/` for fleet-wide conventions:

- FastMCP 3.4+ portmanteau pattern with `operation` enum
- Structured dict responses with `success`, `message`, `data`
- Pydantic v2 (`model_dump()`, not `.dict()`)
- Ruff linting + Biome for TypeScript
