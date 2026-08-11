# Local LLM MCP Server — User Interaction Guide

Welcome! This guide teaches you how to operate the Local LLM MCP Server end to end: discovering models, generating text, managing GPU memory, supervising inference engines, working with the AI gateway, and recovering from failures. The server manages local engines (Ollama, LM Studio, vLLM, llama.cpp) and 28 cloud providers through one unified MCP surface of 13 portmanteau tools.

Everything below is written as tutorials. Work through them in order; each builds on the previous.

## Lesson 1: Hello, Server

Before anything else, confirm the server is alive and see what it exposes. Two discovery calls are the ground truth for every session:

```
llm_health(operation="health_check")
llm_help(operation="list_tools")
```

What you should do with the responses:

1. Read `status` and `registered_tools` from the health check. If the server is healthy you will see the 13 portmanteau tool names.
2. From `list_tools`, note the exact operation enums of the tools you plan to use. The operation lists in the tool schemas are authoritative — do not assume names from memory.
3. If the health check reports a degraded service, run `llm_health(operation="provider_check")` to see which provider or engine is at fault before doing anything else. A server that is alive but has a dead Ollama engine behaves very differently from a server whose tools failed to register.

A healthy response looks like:

```
{"success": true, "status": "healthy", "uptime_seconds": 1234, "registered_tools": ["llm_health", "llm_models", ...]}
```

## Lesson 2: Discovering What Models Exist

Models live in three places, and each has a different discovery tool:

| Where | Discovery call | What it returns |
|-------|---------------|-----------------|
| Server registry (all providers) | `llm_models(operation="list_models")` | Registered aliases with provider, quantization, context length |
| Ollama engine | `llm_ollama(operation="list_models")` | Tags served by Ollama on 11434 |
| LM Studio | `llm_lmstudio(operation="list_models")` | Models served by LM Studio on 1234 |
| vLLM server | `llm_vllm(operation="list_models")` | Models served by the running vLLM container |
| HuggingFace Hub | `llm_huggingface(operation="search_models", query="...")` | Remote models you can download |

Tutorial: a user asks "what can you run?"

1. `llm_models(operation="list_models")` — the registry snapshot.
2. `llm_health(operation="provider_check")` — which engines are reachable *right now* (an engine may be registered but stopped).
3. `llm_ollama(operation="list_models")` — the actual tags on the Ollama disk (these can differ from the registry if tags were pulled outside the server).
4. Summarize: for each engine, list reachable models with a one-line suitability note (size, modality, speed).

Never answer "what can you run" from memory. The discovery calls above are cheap and authoritative; use them.

## Lesson 3: Generating Text

The core operation is `llm_generation(operation="generate_text", model=..., prompt=...)`.

### Choosing parameters

- `temperature`: 0.2-0.4 for code and structured output; 0.7 for balanced conversation; 0.8-1.1 for creative writing.
- `max_tokens`: 512 default. Raise to 1024-2048 for essays, reports, or long-form code. The tool truncates at the cap, so an unexpectedly short answer with no stop reason often means the cap was hit.
- `top_p`: leave at 1.0 unless the user has a specific reason; `temperature` is the primary diversity knob.
- `stop`: useful for extraction tasks, e.g. stop at a JSON terminator or a section heading.

### Chat (multi-turn)

Use `chat_completion` with an OpenAI-format `messages` array — never concatenate conversation history into a single prompt string:

```
llm_generation(operation="chat_completion", model="qwen3.6:32b", messages=[
  {"role": "system", "content": "You are a concise coding assistant."},
  {"role": "user", "content": "Write a function to parse CSV."},
  {"role": "assistant", "content": "Here is one..."},
  {"role": "user", "content": "Now add error handling."},
])
```

### Long outputs

For anything over roughly 500 tokens, prefer `stream_generate`. It returns tokens incrementally, avoids client-side timeouts, and lets the user see progress. The return includes a completion reason so you can tell the user whether the model stopped naturally or hit a cap.

### Embeddings

`embed_text` returns dense vectors. Use it for similarity, clustering, and retrieval tasks. The embedding model must be registered and support embeddings (e.g. a sentence-transformers model through the local provider); text-generation-only models return an explicit capability error — do not retry them, switch models.

## Lesson 4: Model Lifecycle

### Registering a model alias

`llm_models(operation="register_model", name="my-alias", provider="ollama", base_url="http://localhost:11434")` creates an alias so you can reference a model by a short name across tools. `get_model_info` shows the alias details; `update_model` changes provider/base_url/quantization without re-creating.

### Pulling a model (Ollama)

If `list_models` shows a tag is missing, `llm_ollama(operation="pull_model", model="<tag>")` downloads it. Large models take minutes. The tool returns progress; do not report completion until it returns `success: true`. If the user interrupts, the download is resumable by re-issuing the same pull.

### Downloading weights (HuggingFace)

`llm_huggingface(operation="download_model", model_id="owner/repo")` persists weights to the local cache. Gated repos need `HF_TOKEN`; the error message tells the user exactly what to set. After download, register the local path as a model alias so generation tools can use it.

### Unloading to free VRAM

`llm_ollama(operation="unload_model", model=...)` evicts a resident model. Verify the effect with `llm_gpu(operation="get_status")` — VRAM should drop.

### Deleting is destructive

`llm_ollama(operation="delete_model", model=...)` permanently removes a tag; `llm_models(operation="unregister_model", ...)` removes an alias (never the weights). Both require explicit user confirmation. State what will be gone before calling.

## Lesson 5: GPU and VRAM Management

The host has an RTX 4090 with 24 GB VRAM. The two tools are `llm_gpu` (GPU state) and `llm_engine` (engine processes).

### Reading GPU state

`llm_gpu(operation="get_status")` returns per-GPU utilization, temperature, VRAM used/free, and which processes hold VRAM. Read this before any large-model load and before starting a second engine.

### The 24 GB budget

A 30B Q4 model needs roughly 20-21 GB. Ollama with a 32B Q4 is similarly heavy. Running both concurrently will thrash. The fleet's layout acknowledges this: while the llama.cpp server holds Glimmer 30B, Ollama is typically stopped (the truncating proxy on 11435 and llama-server on 11439 share the card).

### Freeing memory

1. `llm_gpu(operation="get_status")` — see what is resident.
2. `llm_engine(operation="status")` — see which engines hold memory.
3. Unload idle models: `llm_ollama(operation="unload_model", ...)` or `llm_engine(operation="unload_model", ...)`.
4. `llm_gpu(operation="clear_memory")` — defragment (PyTorch cache clear; safe between workloads).
5. Reload the target model and confirm headroom.

### Optimizing

`llm_gpu(operation="optimize")` applies the configured memory optimization profile. Run it after heavy workloads before switching tasks.

## Lesson 6: Engine Supervision

`llm_engine` manages the *processes*: Ollama and the natively compiled llama.cpp server. `status` returns process state, PID, port, uptime, VRAM, and loaded models per engine.

- **Engine down** → `llm_engine(operation="start", engine="ollama")` (or `"llama-cpp"`).
- **Engine wedged** → `llm_engine(operation="stop", engine=..., force=True)` then start again.
- **Preload** → `llm_engine(operation="start", engine="ollama", model="qwen3.6:32b")` boots the engine with the model resident, cutting first-token latency.

vLLM is supervised separately through `llm_vllm` because it runs as a Docker container: `start_server` / `stop_server` / `get_server_status` / `get_config` / `update_config`. Use vLLM for batch and throughput workloads; it is not meant for one-off interactive calls.

## Lesson 7: Provider Health and the Circuit Breaker

`llm_health(operation="provider_check")` probes every configured provider with a 60-second cached liveness check. The circuit breaker opens after 3 consecutive failures and enforces a 60-second cooldown.

Why this matters:

- A failed call during cooldown returns immediately with `provider_unreachable` even if the engine has since recovered. Do not retry blindly — wait out the cooldown, fix the root cause, then re-probe.
- The response includes `latency_ms` and `model_count` per provider, which tells you *why* something is slow before you blame the model.
- The same probes power the web dashboard (`/api/v1/health`) and the diagnostics endpoint (`/api/v1/diagnostics`, which forces a fresh probe).

Tutorial: a generation call fails. Do this:

1. Read the error: `error_type` tells you the category (`provider_unreachable`, `auth`, `no_gpu_memory`, `validation`).
2. `llm_health(operation="provider_check")` — is the engine reachable? Is the circuit breaker open?
3. `llm_engine(operation="status")` — is the process running?
4. Fix the actual cause: start the engine, ask for the missing key, free VRAM.
5. Re-run the original call. Report what was wrong, not just that you fixed it.

## Lesson 8: The AI Gateway

The gateway is an OpenAI-compatible endpoint at `http://127.0.0.1:10833/v1/chat/completions`. Any OpenAI SDK client can use it; provider selection is by header or model prefix.

```
client = OpenAI(base_url="http://127.0.0.1:10833/v1", api_key="sk-any")
client.default_headers["x-lightport-provider"] = "deepseek"
client.chat.completions.create(model="deepseek-chat", messages=[...])
```

Supported local providers: Ollama, LM Studio, vLLM (auto-detected). Cloud providers: Anthropic, Azure, Bedrock, Cohere, DeepInfra, DeepSeek, Featherless, Fireworks, Gemini, Groq, Hyperbolic, Lepton, Mistral, Modal, Nebius, Novita, OpenAI, OpenRouter, Perplexity, Replicate, SambaNova, SiliconFlow, Together, xAI (Grok), Anyscale.

Rules of engagement:

- Prefer the MCP tools for one-off generation; use the gateway when a client (a script, another fleet server, a notebook) needs an OpenAI-compatible endpoint.
- Per-call provider selection via header is always safe and does not mutate configuration.
- Gateway provider configuration changes belong in the Settings page / config; do not mutate them as a side effect of a single call.

## Lesson 9: Multimodal Work

`llm_multimodal` covers `analyze_image`, `generate_image`, `compare_images`.

- `analyze_image(image_path=...)`: pass a filesystem path or base64 `image_data`. Only multimodal models (Glimmer, Gemini-class, GPT-4o-class) support this; a text-only model returns an explicit capability error.
- `generate_image(prompt=..., negative_prompt=..., steps=..., width=..., height=...)`: runs the local diffusion pipeline. Default 20 steps at 512x512 is a good starting point; raise steps to 40-50 for quality, and note the longer runtime.
- `compare_images(image_path_a=..., image_path_b=...)`: similarity assessment — useful for verifying that generated images match a reference.

Tutorial: "generate a hero image for my blog post about Alpine hiking":

1. Confirm the diffusion pipeline/model is registered (`llm_models(operation="list_models")`, look for an image-capable model).
2. `generate_image(prompt="Alpine hiking trail at sunrise, dramatic peaks, warm golden light, high detail", negative_prompt="watermark, text, low quality", steps=40, width=1024, height=1024)`.
3. Return the saved path and offer variations (different seed, aspect ratio) or `compare_images` against a reference.

## Lesson 10: Fine-tuning with LoRA Adapters

`llm_finetuning` manages LoRA adapters: `lora_load_adapter`, `lora_unload_adapter`, `lora_list_adapters`.

Tutorial: attach a specialized adapter to a base model.

1. `llm_finetuning(operation="lora_list_adapters")` — what adapters exist.
2. `llm_finetuning(operation="lora_load_adapter", model_name="<base model>", adapter_path="<path to adapter dir>")` — attach it.
3. Generate with the base model — the adapter is now active.
4. `lora_unload_adapter` to detach when done, and confirm with `lora_list_adapters`.

Heavy training (DoRA, sparse, QLoRA-evolved, unsloth) lives in optional tool modules registered only when their dependencies are installed. If a user asks for training and the module is absent, say so explicitly and point to the docs rather than pretending.

## Lesson 11: LM Studio and LM Link

`llm_lmstudio` controls the LM Studio engine: `list_models`, `load_model`, `unload_model`, `eject_model`, and the special `link_status`.

`link_status` probes LM Link — the Tailscale mesh for remote LLM access — via `lms link status --json`. The result includes peers, per-peer loaded models, link state, and preferred device. Use cases:

- The user wants inference on a remote machine with a better GPU: check peers, pick the peer, route through LM Studio.
- The dashboard shows `lm_link` health: the same probe backs `/api/v1/health`.

If the link is down, report whether the problem is the local LM Studio app, the tailnet, or the remote peer. Cross-repo: tailscale-mcp (port 10821) owns LM Link enable/disable and device naming; this server is the read side.

## Lesson 12: Health Metrics and Logging

- `llm_health(operation="get_metrics")` — tokens per second, latency percentiles, VRAM utilization, request counters.
- `llm_health(operation="collect_metrics")` — a fuller snapshot for analysis.
- `llm_health(operation="set_log_level", level="DEBUG")` — runtime verbosity without restart. Use DEBUG to chase a failing operation, then return to WARNING/INFO.
- `llm_health(operation="system_info")` — hardware, OS, Python environment.
- `llm_health(operation="service_status")` — supervised engine processes.

## Lesson 13: Recovering from Common Failures

### Model not found

Run `list_models` and search the returned ids. If it is a HuggingFace model, `search_models` + `download_model` + `register_model` brings it online. Never claim a model is absent without checking both the registry and the engine's tag list.

### Engine unreachable

`provider_check` → `llm_engine(operation="status")` → `llm_engine(operation="start", engine=...)`. For vLLM: `llm_vllm(operation="get_server_status")` then `start_server`.

### Authentication errors

The error names the missing key. Keys live in `.env.example` (repo root) and are loaded from the environment; the Settings page shows which are set. Gated HuggingFace models additionally need `HF_TOKEN`.

### Out of memory

`llm_gpu(operation="get_status")` → unload idle models → `clear_memory` → retry with a smaller quant or shorter context. If the user insists on running two large engines simultaneously on 24 GB, warn clearly about swapping and let them choose.

### Weird or truncated output

Check `max_tokens` first (truncation), then `temperature` (too high for the task), then the model itself (a 7B general model will not match a 30B agentic model on complex tasks). Report the likely cause with the evidence.

## Lesson 14: Comparing Models for a Task

Users frequently want "which model is best for X". A structured comparison beats an opinion:

1. Pick 3 candidate models from `list_models` (e.g. a small fast tag, a medium balanced tag, a large quality tag — or the same model at two quantizations).
2. Run the same prompt through each with identical `temperature` (0.2 for deterministic tasks) and equal `max_tokens`.
3. Collect per-run latency from the response (the server reports timing) and note output quality differences.
4. Also consider VRAM: check `llm_gpu(operation="get_status")` between loads, and only run models that fit simultaneously — otherwise run sequentially and unload between.

Present a small table: model, latency, quality verdict, VRAM used, recommendation for the specific task. Never rank models you did not actually run.

## Lesson 15: Batch Workloads and Benchmarks

For throughput-oriented work (evaluations, dataset labeling, bulk summarization):

1. Start the vLLM server: `llm_vllm(operation="start_server")` with the serving config (tensor parallelism, max-model-len tuned for the workload).
2. Route requests through the gateway: `base_url=http://127.0.0.1:10833/v1`, `x-lightport-provider: vllm`.
3. Monitor `llm_vllm(operation="get_server_status")` and `llm_gpu(operation="get_status")` during the run.
4. Collect `llm_health(operation="get_metrics")` after — tokens/second and latency percentiles are the honest numbers.
5. Stop the server when done: `llm_vllm(operation="stop_server")`.

Note: `stream_generate` also works for batch MCP usage and avoids token caps, but vLLM is the correct tool when the workload is large and OpenAI-compatible.

## Lesson 16: Embeddings and Retrieval

`llm_generation(operation="embed_text", model="<embedding-model>", text=[...])` produces dense vectors. Tutorial: build a mini retrieval index over a set of documents.

1. Chunk the documents into paragraphs.
2. Embed all chunks: `embed_text` accepts a list of texts and returns aligned vectors.
3. Embed the query.
4. Rank chunks by cosine similarity and return the top k as context.
5. Chain into `chat_completion` with the retrieved context for a grounded answer.

The local embedding model must be registered first. If the user's chosen model does not support embeddings, the error tells you — switch to a sentence-transformers class model rather than retrying.

## Lesson 17: Cloud Provider Recipes

Cloud providers need API keys in the environment and are reached through the gateway or the cloud adapters. Common recipes:

### Anthropic (Claude)

```
llm_generation(operation="chat_completion", model="claude-sonnet-4-6", messages=[...])
```

or via gateway: `x-lightport-provider: anthropic`. Requires `ANTHROPIC_API_KEY`. Default model configurable via `ANTHROPIC_MODEL`.

### DeepSeek

Gateway: `x-lightport-provider: deepseek`, model `deepseek-chat` (or `deepseek-reasoner` for chain-of-thought tasks). Requires `DEEPSEEK_API_KEY`.

### Google Gemini

`llm_google_cloud(operation="generate_content", ...)` or gateway `x-lightport-provider: gemini`. Requires `GEMINI_API_KEY` (or Vertex service account). Gemini-class models also accept image input — use `llm_multimodal` for image analysis with a multimodal model.

### OpenAI

Gateway `x-lightport-provider: openai`, models `gpt-4o` class. Requires `OPENAI_API_KEY`.

### OpenRouter (aggregator)

One key, many models: `x-lightport-provider: openrouter`, model id of the target model on OpenRouter.

When a user asks for a cloud model, check the key is configured first (the Settings page shows which providers have keys). An `auth` error means the key is missing or invalid — tell the user exactly which variable to set, then retry after they confirm.

## Lesson 18: System Monitoring as a Service

The server is also a monitoring surface. Tutorial: report a "GPU and fleet health snapshot".

1. `llm_health(operation="system_info")` — CPU, RAM, disk, OS.
2. `llm_gpu(operation="get_status")` — utilization, temperature, VRAM.
3. `llm_health(operation="service_status")` — engine processes and ports.
4. `llm_health(operation="provider_check")` — local + cloud reachability.
5. `llm_lmstudio(operation="link_status")` — LM Link peers (if relevant).
6. Assemble one summary: hardware health, GPU state, engine state, provider state, and anything needing attention.

This mirrors the web dashboard's KPI cards, so a CLI user gets the same picture in text.

## Lesson 19: Fine-Grained Generation Controls

Beyond the basics, the generation tools expose controls worth knowing:

- `stop` sequences: end generation at structural boundaries. For JSON output, stop at `\n}\n` style terminators; for lists, stop at the end marker. The tool reports the stop reason so you can verify.
- Deterministic runs: set `temperature=0` and a fixed `seed` (engine-dependent) to reproduce outputs for tests.
- Multi-turn refinement: generate a draft, then feed it back through `chat_completion` with instructions ("shorten by half", "make it more formal", "add citations"). Iterative refinement beats one-shot prompting for quality.
- System prompts: put task framing in a `system` message (chat) rather than prepending to the user content — models respect the role boundary and you avoid prompt-contamination when the user content is untrusted.

## Lesson 20: Working Around Engine Quirks

### Ollama keeps a model resident

Ollama evicts models lazily. After a big load, unload explicitly when done (`llm_ollama(operation="unload_model", ...)`) — otherwise the next large load may fail with VRAM pressure even though the user "stopped using" the first model.

### llama.cpp servers are long-lived

The llama.cpp server (port 11439, fronted by the truncating proxy on 11435) is built for long sessions with a specific model. Do not stop/restart it casually — a restart re-loads 20+ GB into VRAM and takes minutes. Check `llm_engine(operation="status")` before touching it and tell the user the restart cost.

### LM Studio and LM Link

LM Studio serves whatever is loaded in its UI; `load_model`/`unload_model` change the loaded model but the app must be running. `link_status` tells you whether the mesh is up before you promise remote inference.

### vLLM container lifecycle

`start_server` can take 30-60 seconds (container start + model load). Poll `get_server_status` rather than assuming readiness; the tool reports when the HTTP endpoint is actually serving.

## Lesson 21: Multi-Step Autonomous Workflows

Combine lessons into full tasks. Example: "Set up and demonstrate Qwen for a writing task."

1. `llm_health(operation="health_check")` — server alive.
2. `llm_ollama(operation="list_models")` — is the tag present?
3. If missing: `llm_ollama(operation="pull_model", model="qwen3.6:32b")` — report progress honestly.
4. `llm_gpu(operation="get_status")` — VRAM headroom before load (unload anything idle).
5. `llm_engine(operation="start", engine="ollama", model="qwen3.6:32b")` or `llm_ollama(operation="load_model", ...)`.
6. `llm_generation(operation="chat_completion", ...)` with a system prompt for the writing task.
7. Refine via chat follow-ups.
8. At session end: `llm_ollama(operation="unload_model", ...)`, report what is left running.

Every step is observable, every result is verified, and the user is told what is running at the end. That is the pattern for all multi-step work on this server.

## Lesson 22: Logs, Metrics, and Forensics

When something subtle breaks (intermittent failures, slow first tokens, circuit breaker trips):

1. `llm_health(operation="set_log_level", level="DEBUG")` — capture detail without restart.
2. Reproduce the failure once.
3. `llm_health(operation="get_metrics")` — request counters, latency percentiles, VRAM utilization at the time of failure.
4. `llm_health(operation="collect_metrics")` — full snapshot.
5. `llm_health(operation="provider_check")` — provider state with circuit breaker status.
6. Set log level back to WARNING when done.

The same data lands in the dashboard's Logging page and the backend logs (`logs/` directory), so cross-reference those when the failure happened outside an MCP call.

## Lesson 23: Interop with Other Fleet Servers

The gateway makes this server the fleet's LLM endpoint. Other servers and agents connect with any OpenAI SDK:

- `openai`-compatible chat for general inference (headless agents, notebook scripts).
- `embed_text` output for RAG pipelines in other repos (the embeddings are standard vectors, consumable anywhere).
- Provider health data via `GET /api/v1/health` for monitoring dashboards (the `providers` and `lm_link` keys are the contract).

If a fleet peer reports "LLM not reachable", the first checks are: is the gateway up (`GET http://127.0.0.1:10833/health`), which provider is it trying (`x-lightport-provider`), and is that key set. You can run all three from here.

## Lesson 24: Performance Tuning Quick Reference

| Symptom | First move | Second move |
|---------|-----------|-------------|
| Slow first token | Model cold — preload via engine `start` with model | Check VRAM fragmentation, `clear_memory` |
| Slow throughput | Not using vLLM | Start vLLM container for the workload |
| High VRAM, low utilization | Multiple resident models | Unload idle models |
| High temperature output for code | `temperature` too high | Set 0.2-0.4, retry |
| Truncated answers | `max_tokens` cap | Raise cap, or `stream_generate` |
| Context-limited on long docs | Model context window | Use chunked embedding + retrieval (Lesson 16) |
| Odd outputs from a chat | System/user role confusion | Put framing in system role |

## Frequently Asked Questions

**Q: Can I run Ollama and the llama.cpp Glimmer server at the same time?** Not comfortably — both want 20+ GB of 24 GB VRAM. The fleet runs Glimmer on the llama.cpp server and stops Ollama meanwhile. `llm_engine(operation="status")` shows the current layout.

**Q: What is the truncating proxy on 11435?** The front door for all fleet traffic. It trims oversized requests, pins the user message, and whitelists tools before forwarding to the llama-server on 11439. Do not bypass it for fleet-facing traffic; direct calls are fine for local experiments.

**Q: Why does the gateway need an api_key I never configured?** The gateway is OpenAI-compatible, so clients must send something; any non-empty string works locally. Provider authentication happens server-side via the provider keys.

**Q: My model call says `provider_unreachable` but Ollama is open.** Check the circuit breaker state in `provider_check` — it may be in cooldown after earlier failures. Also confirm the base URL (`OLLAMA_BASE_URL`), especially if Ollama was moved to a non-default port.

**Q: How do I add a new model permanently?** Pull/download it (Ollama/HF), then `register_model` with the provider and defaults. The registry persists across restarts; engine processes do not.

**Q: Are deleted models recoverable?** No. `delete_model` removes the tag permanently; re-pull to restore. `unregister_model` only removes the alias — weights stay.

**Q: Where do I see what the server is doing right now?** Dashboard at `http://localhost:10832` (KPIs, provider health, LM Link), backend at `http://localhost:10833` (`/api/v1/health`, `/api/v1/diagnostics`), or `llm_health(operation="service_status")` in chat.

**Q: What happens when I shut the server down?** `llm_health(operation="shutdown", confirm=True)` runs the graceful shutdown path: cleanup of state, unload of supervised models where possible, and process exit. Engines it did not start (e.g. a manually launched Ollama) are left alone. Always warn the user about in-flight generations before shutting down.

**Q: The dashboard shows a provider red but the MCP call works. Which is right?** The dashboard reads the cached provider health (60s TTL) and the MCP call may have just succeeded after a recovery. Force a fresh probe with `llm_health(operation="provider_check")` — the tool's `force` semantics bypass the cache. If the tool agrees the provider is healthy, the dashboard will catch up on its next poll.

**Q: Can I route a specific request to a specific cloud provider without changing defaults?** Yes. Use the gateway with the `x-lightport-provider` header per request (Lesson 8). Defaults in the config are untouched.

**Q: My embedding vectors seem wrong dimensionally for my index.** Embedding dimension depends on the model. `get_model_info` on the embedding model reports its dimension; re-index with the matching model rather than mixing vectors from different models in one index.

**Q: Do I need a GPU for everything?** No. Cloud providers need no local GPU; local CPU inference works for small models (slowly). The server reports hardware honestly via `system_info` — GPU presence and utilization are facts, not promises. Recommend GPU for large local models, cloud for the rest.

## Closing Notes

The server's job is to make local GPU compute and 28 cloud providers feel like one coherent, observable, reliable LLM service. Every tool returns structured data with recovery guidance; every workflow in this guide ends with the user knowing what is running, why it is running, and what happened. Follow the discovery-first pattern, respect the VRAM budget, confirm destructive actions, and report failures honestly — that is the entire operating manual.

1. **Discover before assuming** — `list_tools` / `list_models` / `provider_check` are cheap; use them.
2. **Check health first** — one `health_check` at session start prevents a cascade of confused failures.
3. **Manage VRAM actively** — unload what you are not using; check `get_status` before loading.
4. **Match model to task** — small/fast for interactive, large for complex, vLLM for batch.
5. **Confirm destructive actions** — deletes, unregister, shutdown need explicit user consent.
6. **Be honest about failures** — relay `error`, `error_type`, and `recovery_options`; never fake success.
7. **Clean up at session end** — unload models you loaded, stop engines you started, and tell the user what is left running.
8. **Use the right surface** — MCP tools for interactive work, gateway for OpenAI-compatible clients, dashboard for human inspection.

## Where to Go Next

- `llm_help(operation="list_tools")` — live tool catalog.
- `llm_help(operation="get_tool_help", ...)` — per-tool documentation.
- `llm_help(operation="search_tools", query="...")` — find functionality by keyword.
- Web dashboard at `http://localhost:10832` — KPIs, provider health, GPU telemetry, chat, settings.
- API backend at `http://localhost:10833` — `/health`, `/api/v1/health`, `/api/v1/diagnostics`, `/v1/chat/completions`.
- Repo docs: `docs/` (CONFIGURATION.md, TOOLS.md, TROUBLESHOOTING.md, GLIMMER.md) and `llms-full.txt`.

The Local LLM MCP server is the fleet's GPU control plane. Operate it deliberately, keep the user informed of what is running and why, and every workflow above will succeed.
