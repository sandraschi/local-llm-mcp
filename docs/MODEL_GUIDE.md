# Local Model Fleet Guide (4090 + 16 GB cards)

What runs where, why Ollama is the default engine, why vLLM sits out, and
which models proved healthy on 2026-10-24 (cold-load smoke test of every
installed Ollama model: load + tiny completion, unload after).

## Engine comparison

| Engine | Strengths | Weaknesses | Fleet role |
|---|---|---|---|
| **Ollama** (`:11434`) | Zero-config `ollama run/pull`; OpenAI-compatible API; GGUF + safetensors; auto VRAM eviction; every fleet tool targets it | Single-model-at-a-time mindset; newest quant formats lag upstream llama.cpp (kquant); no multi-user scheduling | **Default.** Interactive chat, agents, all MCP tooling |
| **LM Studio** (`:1234`) | Best desktop GUI; tries any GGUF in 2 clicks; built-in OpenAI server; good for evaluation | Manual (GUI-first, weak automation); one model at a time; no fleet tooling beyond list/load/eject | Evaluation + trying models by hand; provider in the gateway |
| **Native llama.cpp server** (`:11439`, fronted by `:11435` proxy) | Newest formats first (runs Glimmer kquant that Ollama cannot); DFlash speculative decoding; full control | Hand-rolled ops (compile from source, manual VRAM management); one tenant per GPU | **Glimmer 30B only** — the one model Ollama cannot run |
| **vLLM** (docker compose present, **not running**) | Throughput king: continuous batching, prefix caching, PagedAttention for many concurrent users | Reserves ~90% VRAM by default (`VLLM_GPU_MEMORY_UTILIZATION=0.9` in our compose); fights Ollama/Glimmer for the same 24 GB; ops complexity (Docker, V1 engine flags, FlashInfer); our workload is single-user interactive + agents, not multi-tenant serving | Parked. It earns its place the day we serve concurrent users or batch jobs where throughput beats latency. Until then it would just evict everything else. |
| text-generation-webui (oobabooga) | Historical; huge extension ecosystem | Unmaintained for current models; superseded by Ollama/LM Studio | Dead. Only mentioned because old docs still cite it |
| MLC / KoboldCpp / serverless APIs | Niche strengths (mobile/edge, roleplay) | No fleet integration | Not used |
| Cloud via AI gateway (`:10833`, 28 providers) | Frontier models, no VRAM limit | $$$ (100-300x local per the doctrine), data leaves the box | Overflow: reasoning the local fleet demonstrably cannot do |

**History in one paragraph:** the fleet started on oobabooga (2023), moved
to Ollama as the zero-config standard (2024), added LM Studio as the manual
evaluation bench, and in 2026 added a native llama.cpp lane for one reason:
Meta's Glimmer uses kquant, which no released Ollama understands. vLLM has
been evaluated twice and parked twice — same reason both times: on a single
shared 4090 it is a throughput engine with nothing to parallelize, and its
VRAM reservation evicts the models you actually chat with.

## What fits: RTX 4090 (24 GB, tested 2026-10-24)

All of the below cold-loaded and answered on this box. Sizes are installed
footprints; keep ~2 GB headroom for context.

| Model | Size | From | Type | Good for | Tool use | Multimodal |
|---|---|---|---|---|---|---|
| gemma4:12b (+`-fixed`) | 8.0 GB | Google | Dense 12B | Daily chat, fast agents | Yes | No (text only) |
| gemma4:e4b / e2b | 9.6 / ~5 GB | Google | MoE | Chat, long-ish context | Yes | No |
| gemma4:26b | ~16 GB | Google | Dense 26B | Stronger chat, slower load (~40 s cold) | Yes | No |
| qwen3.6-35b-a3b (official + Unsloth UD-Q4) | ~20 GB | Alibaba | MoE 35B / 3B active | Reasoning + coding, best quality-per-GB here | Yes | No |
| qwen3.6 / qwen3.6:27b | ~17-20 GB | Alibaba | Dense | Coding, general | Yes | No |
| qwen3.8:27b / qwen3.8-tools | ~17 GB | Alibaba | Dense 27B | Coding; `-tools` tuned for function calls | Yes (tools variant) | No |
| mistral-small:24b | ~14 GB | Mistral AI | Dense 24B | Generalist, good code | Yes | No |
| gpt-oss:20b | ~13 GB | OpenAI (Apache 2.0) | MoE 21B / 3.6B active | Reasoning, agentic loops | Yes | No |
| deepseek-r1:8b | ~5 GB | DeepSeek (distill) | Reasoning distill | Cheap reasoning traces | Limited | No |
| qwen3.5-9b-deepseek-distill | ~6 GB | Community distill | Reasoning distill | Cheap reasoning traces | Limited | No |
| llama3.2:3b | ~2 GB | Meta | Dense 3B | Smoke tests, fastest answers | Basic | No |
| llama3.1:8b | ~5 GB | Meta | Dense 8B | Legacy small; superseded but harmless | Basic | No |
| nomic-embed-text | <1 GB | Nomic | Embeddings | RAG vectors (use `/api/embed`, not chat) | N/A | N/A |
| Bonsai-27B (MobiusDevelopment) | ~16 GB | Community | Dense 27B | Experimental flavors | Varies | No |
| muse-glimmer-* (4 Ollama variants) | ~18-30 GB | Meta (fleet GGUFs) | Dense 30B | The house model; canonical instance runs native on `:11439` | Yes | Vision (native lane) |

**Removed 2026-10-24 (all failed cold-load smoke tests):** `mistral-nemo:12b`
(hang, even after clean re-pull), `Qwopus3.8-27B-Flash` (HTTP 500),
`Qwopus3.6-27B-Coder-MTP` (hang), `deepseek-r1:32b` (empty output, 2 attempts).
~64 GB reclaimed. Pattern: exotic third-party GGUF uploads rot first —
prefer official `qwen*` / `gemma*` / `mistral-*` library names and the
Unsloth account; treat random `hf.co/<user>/` uploads as guilty until tested
(which is exactly what the Models page Test button now does).

## What fits: 16 GB cards (guidance, not tested here)

Rule: model file + ~2 GB context headroom must stay under ~14 GB resident.

- **Safe:** dense ≤ 13B in Q4/Q8 (`gemma4:12b` 8 GB, `mistral-nemo:12b`-class
  7-8 GB, 8B models in Q8 ~9 GB), MoE with small files (≤ 12 GB),
  embeddings + rerankers (tiny).
- **No:** 70B anything, dense 32B in Q4 (~20 GB), 35B MoE Q4 (~20 GB),
  `gemma4:26b`, Glimmer 30B.
- **Instead:** run the small half locally, route overflow to the cloud leg of
  the AI gateway (`:10833`) — that split is what the gateway exists for.

## Lifecycle: where to click

- **Models page** (`:10832/models`, this repo): Ollama library section —
  installed table with sizes, per-model **Test** (cold-load smoke) and
  **Remove**, plus a pull box with progress. No shell needed.
- **MCP tools** (`llm_ollama`: list/pull/delete/load/unload,
  `llm_lmstudio`: list/load/unload/eject, `llm_engine`: supervision) —
  same operations for agents.
- **LM Studio models** are managed in LM Studio itself (its catalog, its
  downloads); the fleet only lists/loads/ejects them.

## Verifying this doc

- Engine claims: `GET /api/v1/engines` (llama-server, proxy, Ollama state).
- Model health right now: Models page → Test per model, or rerun the sweep
  (`C:\Users\sandr\AppData\Local\Temp\opencode\run-ollama-sweep.ps1` pattern:
  cold load + `Reply with exactly: OK` + unload, 100 s timeout).
- 2026-10-24 sweep log: 33 entries probed, 25 OK, 4 removed (above),
  3 cloud refs skipped, 1 embed model tested via `/api/embed`.
