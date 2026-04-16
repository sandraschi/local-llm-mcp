# Local LLM MCP Server - Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-04-15

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
