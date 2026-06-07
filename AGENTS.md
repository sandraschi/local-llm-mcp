# local-llm-mcp Agent Context

Optional fleet **LLM hub** — see [docs/PURPOSE.md](docs/PURPOSE.md). Not the default Ollama sampling backend for peer MCPs.

## Quick ref

```powershell
uv sync
uv run llm-mcp
uv run pytest tests\ -q
just lint
```

Ports: dashboard `10832` / API `10833`.

## Manifests

- `llms.txt` + `llms-full.txt` are **hand-curated** (fleet standard).
- Do **not** overwrite with naive auto-generation (old runs pulled notepadpp-mcp template docs and `.git/config`).
- Regenerate with **llm-txt-mcp** `quality_mode=true`, then manually review MCP tool tables.

## Stale template docs

`docs/notepadpp/`, `docs/repository-protection/` (partial), and some `docs/serena/` files are copied from other repos — ignore for LLM context unless explicitly updating them.
