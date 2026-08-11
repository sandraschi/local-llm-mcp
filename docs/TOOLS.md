# Tool Reference

## Portmanteau Tools

### `llm_health`

Health monitoring, system info, metrics, and provider checks.

| Operation | Description | Required params |
|-----------|-------------|-----------------|
| `health_check` | Overall system health | — |
| `provider_check` | Check provider connectivity | — |
| `system_info` | Detailed system information | — |
| `service_status` | Service status summary | — |
| `get_metrics` | Performance metrics | — |
| `set_log_level` | Change log level at runtime | `level` |
| `shutdown` | Graceful server shutdown | `confirm=true` |

### `llm_models`

Model management and provider operations.

| Operation | Description | Required params |
|-----------|-------------|-----------------|
| `list_models` | List all available models | — |
| `get_model_info` | Model details | `model_id` |
| `register_model` | Register a custom model | `name`, `provider` |
| `ollama_list` | List Ollama models | — |
| `ollama_pull` | Pull model from Ollama | `model` |
| `ollama_delete` | Delete Ollama model | `model` |
| `lmstudio_list` | List LM Studio models | — |
| `vllm_list` | List vLLM models | — |

### `llm_generation`

Text generation, chat completion, and embeddings.

| Operation | Description | Required params |
|-----------|-------------|-----------------|
| `generate_text` | Generate text from prompt | `model`, `prompt` |
| `chat_completion` | Chat completion from messages | `model`, `messages` |
| `embed_text` | Generate text embeddings | `model`, `text` |

### `llm_multimodal`

Image analysis and generation.

| Operation | Description | Required params |
|-----------|-------------|-----------------|
| `analyze_image` | Describe image content | `image` |
| `generate_image` | Generate image from prompt | `prompt` |
| `compare_images` | Compare two images | `image1`, `image2` |

### `llm_finetuning`

LoRA, DoRA, and sparse fine-tuning.

| Operation | Description | Required params |
|-----------|-------------|-----------------|
| `lora_list_adapters` | List installed LoRA adapters | — |
| `lora_load_adapter` | Load LoRA adapter | `adapter_name`, `adapter_dir` |
| `lora_unload_adapter` | Unload LoRA adapter | `adapter_name` |
| `dora_load_model` | Load model with DoRA | `model_name` |
| `dora_train` | Train with DoRA | `model_name` |
| `sparse_load_model` | Load model with sparse FT | `model_name` |
| `sparse_train` | Train with sparse FT | `model_name` |

## Provider-Specific Tools

### `llm_ollama`

| Operation | Description | Required params |
|-----------|-------------|-----------------|
| `list_models` | List Ollama models | — |
| `pull_model` | Pull a model | `model` |
| `load_model` | Load model into memory | `model` |
| `unload_model` | Unload from memory | `model` |
| `delete_model` | Delete a model | `model` |

### `llm_lmstudio`

| Operation | Description | Required params |
|-----------|-------------|-----------------|
| `list_models` | List LM Studio models | — |
| `load_model` | Load model | `model` |
| `unload_model` | Unload model | `model` |
| `link_status` | LM Link peer status | — |

### `llm_vllm`

| Operation | Description | Required params |
|-----------|-------------|-----------------|
| `list_models` | List loaded vLLM models | — |
| `load_model` | Load model | `model_name` |
| `unload_model` | Unload model | — |
| `get_status` | vLLM engine status | — |

### `llm_gpu`

| Operation | Description | Required params |
|-----------|-------------|-----------------|
| `get_status` | GPU status and VRAM | — |
| `clear_memory` | Clear GPU memory | — |
| `optimize` | GPU memory optimization | — |
| `get_health` | GPU health diagnostics | — |

### `llm_huggingface`

| Operation | Description |
|-----------|-------------|
| `search_models` | Search HuggingFace models |
| `search_datasets` | Search datasets |
| `get_model_info` | Model card details |
| `download_model` | Download model weights |
| `whoami` | Current HF auth status |

### `llm_google_cloud`

| Operation | Description |
|-----------|-------------|
| `generate_text` | Gemini text generation |
| `chat_completion` | Gemini chat |
| `generate_content` | Multimodal content generation |
| `list_models` | List Vertex AI models |
| `list_buckets` | List Cloud Storage buckets |

## Legacy Tools (optional)

Enable with `LLM_MCP_ENABLE_LEGACY_TOOLS=true`:

- `list_tools`, `get_tool_help`, `search_tools`
- `get_system_info`, `get_environment`
- `get_metrics`, `health_check`
- `list_models`, `get_model_info`, `ollama_list_models`
