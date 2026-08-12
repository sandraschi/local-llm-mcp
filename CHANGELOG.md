# Local LLM MCP Server - Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [1.0.0] - 2026-08-12

### Added
- **MCPB prompts to fleet 3-4-100 bar**: system.md (3228 words), user.md (4145 words), examples.json (119 entries) - fully rewritten for the current 13-tool portmanteau surface.
- **Real manifest.json**: proper author/repo/entry_point (run_server.py) and the current 13 portmanteau tool schemas; version aligned to 1.0.0.
- **Prefab status card**: `show_status_app` in-chat dashboard card (PrefabApp) registered via `@mcp.tool(app=True)`.
- **Webapp Tools page** (`/tools`): live dynamic tool catalog from new `GET /api/v1/tools` endpoint - never a hardcoded list.
- **Webapp Skills page** (`/skills`): lists server skills and renders SKILL.md via new `GET /api/v1/skills` + `GET /api/v1/skills/{name}` endpoints.
- **Local engine glom-on** in Settings: client-side Ollama/LM Studio/vLLM auto-detection with status dots, provider/model selects (`llm-provider-select`/`llm-model-select`), localStorage persistence (`llm_provider`/`llm_model`), GPU opportunity hint.
- **Tool registration fixes**: gpu_manager (broken async body), moe_tools (dropped FastMCP 3.x-incompatible `stateful=` and `**kwargs`), qloraevolved (*args wrappers -> explicit signatures, removed double registration). 28 tools now register cleanly (was 20 with 3 silent failures).
- **FastMCP 3.4.5 API fixes**: `get_tools()` -> `list_tools()` in main.py and help_tools (compat helper supports both).
- **CORS on HTTP transport**: transport.py now serves via uvicorn.Server on `mcp.http_app()` with fleet CORS middleware (`run_http_async` dropped middleware).
- **T20 print ban** enabled in ruff select; error-path prints in main.py converted to logger calls.
- pyright and ty added to dev dependencies; pyproject description mojibake fixed; `ServerConfig.port` default corrected to 10833; Pydantic v1 `.dict()` calls migrated to `model_dump()`.
- `.git/hooks/pre-commit` materialized; CI ruff gate fixed (tests I001); root junk and .bak dross removed; glama.json tool list refreshed to the 13 portmanteaus; `just mcpb-pack` version corrected to v1.0.0.

## [1.2.3] - 2026-07-29

### Added
- **RAG as a Service**: Optional LanceDB vector store with sentence-transformers embeddings.
  `POST /api/v1/rag/ingest`, `GET /api/v1/rag/search`, `DELETE /api/v1/rag/clear`.
  Opt-in via `uv sync --extra rag`. Returns 501 gracefully when deps missing.
  Includes `skills/rag-expert/SKILL.md` with usage recipes and cross-repo examples.
- **Settings LLM provider detection**: Auto-probes Ollama, LM Studio, vLLM via backend health.
  Shows Detected/Not found per provider with model count. Active provider and model
  selectors persist to localStorage per WEBAPP_SOTA_STANDARDS.md §VI.
- **Dashboard hero section**: Oversized brand header with live status chips
  (model count, server version, uptime). Real GPU telemetry (RTX 4090 VRAM used/total).
  `data-testid` attributes on all KPIs.

### Fixed
- **Models endpoint**: `/api/v1/models` now auto-discovers Ollama/LM Studio models by
  direct probe when ModelService registry is empty — returns 20+ real model names instead of `[]`.
- **Chat generation**: `POST /api/v1/generate` with provider="ollama" no longer returns
  "Provider not found". Falls back to calling Ollama's `/api/chat` directly.
- **API config URLs**: `getConfig()`/`updateConfig()` now hit `/api/v1/config` (was `/config` — 404).
- **GPU telemetry**: Fixed type mapping so dashboard shows real GPU name and VRAM usage,
  not "—".
- **TypeScript build**: Fixed `vision.tsx` missing imports, `Logging.tsx` useRef arg,
  `chat.tsx` unused variable. `tsc --noEmit` and `bun run build` both clean.

## [1.2.2] - 2026-07-29

### Added
- **CORS hardening**: Added `allow_origin_regex` for Tailscale/LAN IPs per fleet standard
- **Session context injection**: `.claude-plugin/plugin.json` + `hooks/hooks.json` for Claude Code,
  `.windsurfrules` for Windsurf, `.github/copilot-instructions.md` for GitHub Copilot
- **Self-termination MCP tool**: `llm_health(operation="shutdown", confirm=True)` for graceful shutdown
- **Automated error logging**: `_error_response()` helper in `tool_utils.py` with `logger.exception()` auto-logging

### Fixed
- **Pydantic v2 migration**: Replaced all deprecated `.dict()` calls with `.model_dump()` (6 files)
- **Bare `except:` in Ollama provider**: Changed to `except Exception` with proper logging
- **Port zombie clearing**: `start.ps1` now clears port 10833 before binding with readiness poll
- **Line length violations**: Fixed E501 in `mcp_servers.py`

## [1.2.1] - 2026-07-06

### Added
- **LM Link awareness**: `llm_lmstudio` tool gains `link_status` operation
  - Probes `lms link status --json` via async subprocess for live peer/model discovery
  - Returns connected peers, their loaded models, link state, and preferred device
  - Graceful fallback with `recovery_options` when `lms` CLI is missing
- **LM Link provider health**: `check_lm_link_status()` in `provider_health.py`
  - 60-second cache TTL with forced refresh support
  - Results exposed in `GET /api/v1/health` and `GET /api/v1/diagnostics` under `lm_link` key
  - Structured `LinkStatus` dataclass: ok, enabled, device_name, peers, peer_count

### Changed
- `llm_lmstudio` portmanteau now supports 4 operations (was 3): list_models, load_model, unload_model, link_status
- `GET /api/v1/health` response includes `lm_link` peer data alongside provider status

## [1.2.0] - 2026-07-03

### Added
- **Provider Health Service** (`services/provider_health.py`)
  - Unified liveness checks for Ollama and LM Studio with 3s connect timeout
  - Result caching with 30-second TTL
  - Circuit breaker: 3 consecutive failures → mark unavailable for 60 seconds
  - LM Studio Docker port-conflict detection (validates content-type + JSON shape on :1234)
- **Provider Health Endpoints**
  - `GET /api/v1/health` — fleet-standard health with provider status
  - `GET /api/v1/diagnostics` — CUA-NSIS smoke test endpoint (tool count, provider status, system info)
  - `GET /v1/gateway/providers/health` — per-provider reachability probe
- **Provider check MCP tool** — `llm_health(operation="provider_check")` returns structured Ollama/LM Studio health
- **Connection hardening** across all local provider code paths:
  - Granular timeouts: 5s connect, 30s read, 300s pull
  - Retry with exponential backoff (1s, 2s, 4s) on connection failures
  - Structured error types: `connection_refused`, `timeout`, `docker_conflict`, `circuit_open`

### Fixed
- **Ollama `unload_model`**: was sending bogus `POST /api/chat` with empty model; now uses correct `POST /api/generate` with `keep_alive: 0`
- **`core/startup.py`**: removed dead import from non-existent `..managers.model_manager` that crashed on import
- **`services/model_service.py`**: fixed unreachable `_initialized = True` after `return`; missing `HTTPException`/`status` imports; duplicate dead `except Exception` block; missing `try:` block in `get_model_info`
- **`services/model_intelligence.py`**: removed extra `}` causing syntax error
- **`gateway/adapters/bedrock.py`**: added missing `import httpx` (F821)
- **`tools/dora_tools.py`**: added missing `import asyncio` (F821)
- **`providers/gemini/provider.py`**, **`providers/perplexity/provider.py`**: fixed `file_path` → `model.get("id", "")` (F821)

### Added
- **Model Orchestration Dashboard** (Vite + React + Tailwind)
  - Unified hub for monitoring and controlling multiple LLM providers.
  - Interactive **Fleet Launcher** for navigating the local MCP ecosystem.
  - Real-time **Engine Analytics** dashboard with GPU/RAM telemetry.
  - Glassmorphism-based premium UI with dark mode support.
- **Live Configuration API** (`/api/v1/config`)
  - Enables browser-side updates to provider URLs and API keys.
  - Persistent storage of configuration directly back to the `.env` file.
  - Nested Pydantic-aware update engine for complex settings objects.

### Fixed
- **Backend Stability**: Resolved critical `ImportError` caused by naming collision between `models.py` and the `models/` directory.
- **Frontend Build**: Fixed TypeScript compilation errors related to `SpeechRecognition` and Vite environment variables.
- **Process Management**: Improved port cleaner in `start.ps1` to handle orphaned backend/frontend instances.
- **Documentation**: Corrected port assignments (10832/10833) across the repository.

## [1.0.1] - 2025-01-08

### Added
- **Google Cloud Portmanteau Tool** (`llm_google_cloud_tool`)
  - Gemini 3.0 Flash (Experimental) support
  - Nano Banana Pro and other latest Gemini models
  - Vertex AI integration for enterprise deployments
  - Google Cloud Storage operations (upload/download/manage)
  - Model deployment to Vertex AI endpoints
  - Dual authentication: Gemini API and Vertex AI
  - Environment variable support: `GOOGLE_CLOUD_TOKEN`, `GOOGLE_CLOUD_PROJECT`, etc.

- **Hugging Face Portmanteau Tool** (`llm_huggingface_tool`)
  - Full gated model support (FLUX, Black Forest Labs models)
  - Dataset download and management
  - Repository operations (create, delete, list)
  - Automatic authentication with `HUGGINGFACE_TOKEN` or `HF_TOKEN`
  - Enhanced error handling for gated content

- **Extensive Multilevel Help System** (10 new tools)
  - `list_available_tools` - 5-level tool discovery (names → expert details)
  - `get_tool_help` - Comprehensive tool documentation
  - `search_tools` - Relevance-scored tool search
  - `get_tool_signature` - Function signatures with metadata
  - `get_workflow_guides` - Complete workflow documentation
  - `get_performance_guide` - Performance optimization strategies
  - `get_troubleshooting_guide` - Comprehensive issue resolution
  - `get_hardware_requirements` - Hardware recommendations and limits
  - `get_quick_reference` - Essential commands and settings
  - `get_integration_guide` - External system integration guides

- **Enhanced GPU Management** (RTX 4090 optimized)
  - Memory fragmentation prevention
  - Advanced memory optimization routines
  - Real-time health monitoring
  - Thermal management guidance

### Changed
- **Tool Count**: Increased from 20 to 31 specialized tools
- **Portmanteau Tools**: Expanded from 8 to 10 consolidated interfaces
- **Documentation**: Updated all system prompts, examples, and manifests
- **Configuration**: Enhanced environment variable support for all providers
- **Architecture**: Improved portmanteau pattern implementation

### Technical Enhancements
- **SOTA Compliance**: Full FastMCP 2.14.1+ compatibility
- **Provider Integration**: Unified config system for all LLM providers
- **Error Handling**: Enhanced structured error responses
- **Performance**: Optimized tool registration and caching
- **Security**: Improved authentication patterns for gated models

## [1.0.0] - 2025-01-07

### Added
- Initial release of Local LLM MCP Server
- FastMCP 2.12+ framework implementation
- Multi-provider LLM support (Ollama, LM Studio, vLLM, OpenAI, Anthropic)
- Portmanteau tool architecture for consolidated operations
- GPU management tools for NVIDIA RTX series
- Basic help and documentation system
- Docker containerization support
- RESTful API and WebSocket interfaces
- Structured logging and monitoring
- MCPB packaging for Claude Desktop integration

### Technical Features
- Dual interface architecture (Stdio + HTTP/WebSocket)
- vLLM 0.10.1.1 integration for high-performance inference
- Advanced fine-tuning support (LoRA, Sparse, DoRA)
- Multimodal capabilities (text, images, audio)
- Real-time system health monitoring
- Comprehensive error handling and recovery

## Version History

### Development Phases
- **Phase 1 (Q3 2024)**: Core MCP server implementation
- **Phase 2 (Q4 2024)**: Multi-provider integration and optimization
- **Phase 3 (Q1 2025)**: Advanced features and enterprise capabilities
- **Phase 4 (Q2 2025)**: Portmanteau tools and extensive help system
- **Phase 5 (Q3 2025)**: Google Cloud and Hugging Face integrations

### Compatibility Matrix

| Component | Version | Status |
|-----------|---------|--------|
| FastMCP | 2.14.1+ | ✅ Compatible |
| vLLM | 0.10.1.1 | ✅ Compatible |
| Python | 3.10+ | ✅ Supported |
| PyTorch | 2.4.0+ | ✅ Supported |
| Transformers | 4.44.0+ | ✅ Supported |

### Migration Notes

#### From v0.x to v1.0.0
- Portmanteau tools replace individual provider tools
- Environment variable configuration now required for providers
- Enhanced error handling may change error message formats
- GPU memory management now automatic

#### Breaking Changes
- Legacy individual tools moved to opt-in only
- Configuration format updated for provider consistency
- Tool signatures enhanced with additional metadata

---

**Legend:**
- ✅ Added feature
- 🔄 Changed behavior
- 🐛 Bug fix
- 📚 Documentation
- 🔒 Security enhancement
- 🚀 Performance improvement

## [1.0.1] - 2026-08-12

### Fixed
- **ty typecheck clean (521 -> 0 diagnostics)**: the CI type gate now actually passes. Systematic fixes (status shadowing in mcp_servers endpoint, transport.py bare ignore) plus ty-sanctioned # ty: ignore[...] comments on legacy optional-dep modules (vllm/torch stubs). CI's continue-on-error step is now a real pass.
- **biome clean (2 a11y warnings)**: Logging.tsx clear-logs modal backdrop is now a real button + dialog role.
- **vllm_v1 provider**: removed dummy-class union for optional vllm imports.
