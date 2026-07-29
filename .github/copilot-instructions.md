## Session Context (Local LLM MCP)

You have access to Local LLM MCP with 8 portmanteau tools for managing local and cloud LLMs: model discovery, text generation, embeddings, multimodal, fine-tuning, GPU telemetry, and LM Studio management with LM Link peer discovery.

**Before starting work:**
1. Check which providers are reachable: `llm_health(operation="provider_check")`
2. List available models: `llm_models(operation="list_models")`
3. Check LM Link for remote LLM peers: `llm_lmstudio(operation="link_status")`

**At end of work:**
- Unload models you loaded if no longer needed
- Verify provider health: `llm_health(operation="provider_check")`
