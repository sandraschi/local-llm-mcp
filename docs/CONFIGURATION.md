# Configuration

## Environment Variables

Set these in your shell or in `claude_desktop_config.json` under `env`.

### Server

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | HTTP port |
| `LOG_LEVEL` | `info` | Log level (debug, info, warning, error) |
| `DEBUG` | `false` | Enable debug mode |

### Authentication

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEYS` | — | Comma-separated API keys for server auth |

### Local LLM Providers

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `VLLM_BASE_URL` | `http://localhost:8000` | vLLM API endpoint |
| `LMSTUDIO_BASE_URL` | `http://localhost:1234` | LM Studio API endpoint |

### Cloud LLM Providers

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `PERPLEXITY_API_KEY` | — | Perplexity API key |

Additional providers (DeepSeek, Groq, xAI, Mistral, OpenRouter, etc.) follow
the same pattern: `<PROVIDER>_API_KEY` and optionally `<PROVIDER>_BASE_URL`.

### Chat Defaults

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_PROVIDER` | `anthropic` | Default chat provider |
| `DEFAULT_MODEL` | `claude-3-opus-20240229` | Default chat model |
| `DEFAULT_TEMPERATURE` | `0.7` | Generation temperature |
| `DEFAULT_MAX_TOKENS` | `2000` | Max tokens per response |
| `CHAT_HISTORY_SIZE` | `1000` | Messages kept in history |

### Caching & Rate Limiting

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_CACHE` | `true` | Enable response caching |
| `CACHE_TTL` | `3600` | Cache TTL in seconds |
| `RATE_LIMIT` | `100` | Requests per minute |
| `RATE_LIMIT_WINDOW` | `60` | Rate limit window in seconds |

## Setting Variables

In `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "llm-mcp": {
      "command": "uv",
      "args": ["--directory", "C:\\path\\to\\local-llm-mcp", "run", "llm-mcp"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```
