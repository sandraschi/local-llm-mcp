# AI Gateway — Lightport-compatible proxy

Provides `POST /v1/chat/completions` that translates OpenAI-format requests to/from 27 LLM provider native formats. Ported from [glama-ai/lightport](https://github.com/glama-ai/lightport).

## Usage

```bash
# Select provider via header
curl http://127.0.0.1:10833/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-lightport-provider: anthropic" \
  -H "Authorization: Bearer $ANTHROPIC_API_KEY" \
  -d '{"model": "claude-sonnet-4-20250514", "messages": [{"role": "user", "content": "Hello"}]}'
```

Or use any OpenAI client — just point the base URL at local-llm-mcp and set `x-lightport-provider`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:10833/v1", api_key="...")
# Set provider header via default_headers
client.default_headers["x-lightport-provider"] = "deepseek"
```

## Supported Providers

| Provider | Header value | Base URL | API key env | Type |
|----------|-------------|----------|-------------|------|
| Ollama | `ollama` | http://127.0.0.1:11434 | — | Local |
| LM Studio | `lmstudio` | http://127.0.0.1:1234/v1 | — | Local |
| vLLM | `vllm` | http://127.0.0.1:8000/v1 | — | Local |
| OpenAI | `openai` | https://api.openai.com/v1 | `OPENAI_API_KEY` | Cloud |
| Anthropic | `anthropic` | https://api.anthropic.com/v1 | `ANTHROPIC_API_KEY` | Cloud |
| DeepSeek | `deepseek` | https://api.deepseek.com/v1 | `DEEPSEEK_API_KEY` | Cloud |
| Gemini | `gemini` | https://generativelanguage.googleapis.com/v1beta | `GEMINI_API_KEY` | Cloud |
| Groq | `groq` | https://api.groq.com/openai/v1 | `GROQ_API_KEY` | Cloud |
| xAI (Grok) | `xai` | https://api.x.ai/v1 | `XAI_API_KEY` | Cloud |
| Mistral | `mistral` | https://api.mistral.ai/v1 | `MISTRAL_API_KEY` | Cloud |
| OpenRouter | `openrouter` | https://openrouter.ai/api/v1 | `OPENROUTER_API_KEY` | Cloud |
| Together | `together` | https://api.together.xyz/v1 | `TOGETHER_API_KEY` | Cloud |
| Fireworks | `fireworks` | https://api.fireworks.ai/inference/v1 | `FIREWORKS_API_KEY` | Cloud |
| Perplexity | `perplexity` | https://api.perplexity.ai | `PERPLEXITY_API_KEY` | Cloud |
| DeepInfra | `deepinfra` | https://api.deepinfra.com/v1/openai | `DEEPINFRA_API_KEY` | Cloud |
| Hyperbolic | `hyperbolic` | https://api.hyperbolic.xyz/v1 | `HYPERBOLIC_API_KEY` | Cloud |
| Novita | `novita` | https://api.novita.ai/v3/openai | `NOVITA_API_KEY` | Cloud |
| Featherless | `featherless` | https://api.featherless.ai/v1 | `FEATHERLESS_API_KEY` | Cloud |
| Nebius | `nebius` | https://api.nebius.ai/v1 | `NEBIUS_API_KEY` | Cloud |
| SiliconFlow | `siliconflow` | https://api.siliconflow.cn/v1 | `SILICONFLOW_API_KEY` | Cloud |
| Lepton | `lepton` | https://api.lepton.ai/v1 | `LEPTON_API_KEY` | Cloud |
| Anyscale | `anyscale` | https://api.endpoints.anyscale.com/v1 | `ANYSCALE_API_KEY` | Cloud |
| Replicate | `replicate` | https://api.replicate.com/v1 | `REPLICATE_API_TOKEN` | Cloud |
| SambaNova | `sambanova` | https://api.sambanova.ai/v1 | `SAMBANOVA_API_KEY` | Cloud |
| Azure OpenAI | `azure` | https://{resource}.openai.azure.com | `AZURE_OPENAI_API_KEY` | Cloud |
| AWS Bedrock | `bedrock` | https://bedrock-runtime.{region}.amazonaws.com | AWS credentials | Cloud |
| Modal | `modal` | https://{app}.modal.run/v1 | `MODAL_API_KEY` | Cloud |
| Cohere | `cohere` | https://api.cohere.com/v2 | `COHERE_API_KEY` | Cloud |

## Endpoints

| Route | Description |
|-------|-------------|
| `POST /v1/chat/completions` | OpenAI-compatible chat completions |
| `GET /v1/models` | List available models per provider |
| `GET /v1/gateway/providers` | List registered provider names |
