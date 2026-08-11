# Muse Glimmer 30B — The Complete Guide

**Source material:** [Meta AI Research blog — Introducing Muse Glimmer (2026-08-10)](https://research.meta.ai/blog/introducing-muse-glimmer-open-agentic-model) ·
[Methodology report](https://research.meta.ai/static/muse-glimmer-methodology) ·
[Model card](https://huggingface.co/meta-models/Muse-Glimmer-30B) ·
[Developer docs](https://dev.meta.ai/docs/muse-glimmer)

**In this fleet:** served natively via llama.cpp on the RTX 4090, supervised by
`llm_engine`, consumed by opencode, aiwatcher, and arxiv-mcp. See
[Engine Operations](#engine-operations-in-this-fleet) below.

---

## 1. What is Muse Glimmer?

Muse Glimmer is Meta Superintelligence Labs' **30-billion-parameter agentic model**
-- the "little brother" of Muse Spark, distilled specifically to run **on your
device**: a Mac or PC with a single consumer GPU, with or without internet.

| Property | Value |
|----------|-------|
| Parameters | ~29.6B (includes vision encoder) |
| License | **Apache 2.0** (weights open) |
| Release | 2026-08-09/10 |
| Modalities | Input: text + image · Output: text |
| Context length | **131,072+ tokens** |
| Architecture | Dense causal transformer + ViT-G/14 perception encoder (~1.8B) |
| Layers | 52 (attention pattern: Local, Local, Local, Global; sliding window 2048) |
| Attention | 32 Q / 2 KV heads (GQA 16:1), head dim 128 |
| FFN | SwiGLU, intermediate 19,968 |
| Positional | RoPE (theta 500,000), local layers only |
| Vocab | 200,000 BPE + 2,048 special tokens |
| Knowledge cutoff | 2026-01-04 |
| Multilingual | 100+ languages |

**Why it exists:** most agent deployments still depend on the cloud. Glimmer is
optimized for **always-on local agent workflows**: personal agents, function
calling, local coding, and LLM-as-a-judge -- running entirely on consumer hardware.

## 2. How it was trained

Three phases (logit distillation from the much larger Muse Spark teacher):

1. **Pre-training** — trained on Muse Spark's outputs using **logit distillation**,
   with a similar data mix as the teacher.
2. **Mid-training** — longer-context, more agent-heavy data with richer reasoning
   traces, alongside organic data.
3. **Post-training** — supervised fine-tuning combined with **on-policy
   distillation and reinforcement learning** across general, reasoning, coding,
   and agentic domains.

## 3. What it can do

- **End-to-end agentic task completion** — strong success rates on DeepSearch QA,
  MCP-Atlas, tau-Bench, and SWE-Bench: work within scaffolds, write and debug
  code, resolve multi-turn requests.
- **Reliable tool use** — precise schema-based function calls across extended
  workflows (native ATEM tool format, parsed into OpenAI-compatible tool calls).
- **Multi-step reasoning** — coherent plans over long horizons.
- **Failure recovery** — diagnoses failed tool calls and retries instead of halting.
- **Multimodal input & reasoning** — screenshots, charts, documents via the
  perception encoder (max 4,096 visual tokens per image).
- **Scaffold compatibility** — works across OpenClaw and other agentic scaffolds.
- **Controllable effort** — reasoning strength: low / medium / high / xhigh
  (set as part of the system prompt: `Reasoning strength: <value>`).
- **Multilingual** — 100+ languages.

## 4. Benchmarks (vs the size class)

From Meta's evaluation (details in the [methodology report](https://research.meta.ai/static/muse-glimmer-methodology)):

| Category | Benchmark | **Muse Glimmer 30B** | Gemma4-31B | Qwen3.6-27B |
|----------|-----------|----------------------|------------|-------------|
| Agentic | MCP-Atlas (public) | **75.5** | 54.2 | 62.5 |
| Agentic | DeepSearch QA | **74.6** | 61.7 | 71.1 |
| Agentic | tau3-Banking | **23.5** | 15.1 | 16.7 |
| Agentic | WildClawBench | **47.6** | 37.6 | 43.2 |
| Agentic | Gaia2 | **43.3** | 36.4 | 40.0 |
| Coding | SWE-Bench Pro | **51.2** | 36.9 | 50.2 |
| Coding | SWE-Bench Verified | 76.0 | 66.6 | **77.2** |
| Coding | SciCode | **43.6** | 43.4 | 39.8 |
| Multimodal | Charxiv Reasoning | **78.8** | 77.7 | 78.4 |
| Reasoning | AIME 2026 | **94.7** | 89.2 | 94.1 |
| Reasoning | IFBench | **77.0** | 76.0 | 70.8 |

Reading: Glimmer wins the **pure-agentic set** decisively (MCP-Atlas +21 over
Gemma, +13 over Qwen) and leads reasoning (AIME 94.7). Qwen3.6-27B counters on
SWE-Bench Verified and OSWorld. Within the fleet the recommendation stands:
**Glimmer = agent/tool-use specialist; Qwen3.6 = coder** if you ever need the
alternative.

## 5. Optimized for local deployment

### Quantization (K-Quant-17GB)

Full precision would need 55+ GB. Meta's 4-bit k-quants shrink the language model
to **under 20 GB**, leaving headroom for KV cache + perception encoder + drafter
inside a 24 GB or 32 GB envelope.

| | Full precision | K-Quant-Dynamic | K-Quant-17GB |
|---|---|---|---|
| Degradation (avg, 15 benchmarks) | — | 0.2% | 1.0% |
| Target hardware | 64 GB VRAM | 32 GB VRAM | **24 GB VRAM** |

### Speculative decoding (DFlash drafter)

A small block-diffusion companion (5 layers, proposes blocks of 16 tokens) that
the main model verifies in parallel -- same output quality, much higher speed:

| GPU | No speculation | With DFlash | Speedup |
|-----|---------------|-------------|---------|
| RTX 5090 | 74.9 tok/s | 233.4 tok/s | **3.1x** |
| Apple M5 Max | 26.6 | 50.2 | 1.8x |
| Apple M4 Max | 23.7 | 37.8 | 1.5x |

The fleet measures ~35-45 tok/s on the RTX 4090 with the drafter active.

## 6. Engine operations in this fleet

```
opencode / fleet servers  -->  :11435 truncating proxy  -->  :11439 llama-server
                                      (trims, pins,                 (Glimmer,
                                       whitelists tools)             131K ctx, full GPU)
```

| Port | Service | Notes |
|------|---------|-------|
| 11435 | Truncating proxy | The front door. Trims system prompt, pins your last message, counts/whitelists tools, tells the model honestly what it can do |
| 11439 | llama-server | Glimmer: kquant-17gb + mmproj + DFlash drafter, `--reasoning on --reasoning-format deepseek --reasoning-budget 1024`, 131072 ctx |
| 11434 | Ollama | Not running while Glimmer holds ~21 GB VRAM |

**Supervision:** `llm_engine(operation="status")` -- processes, ports, VRAM per
engine, loaded models. Start/stop via `llm_engine(operation="start|stop", engine="llama")`.

**Consumers:** opencode (`local-llama/muse-glimmer-30b`), aiwatcher distillation
(`LLM_BASE_URL=...:11435/v1`), arxiv-mcp epistemic jobs
(`ARXIV_MCP_SAMPLING_BASE_URL=...:11435/v1`, model `muse-glimmer-30b`).

**Serving playbook:** `mcp-central-docs/patterns/LLAMA_CPP_NATIVE_MODEL_SERVING.md`
(CUDA build from source, cudart DLLs, the four proxy guardrails, verification checklist).

### The four proxy guardrails (why they exist)

1. **System truncation** -- the fleet instruction wall flips Glimmer into agent
   mode ("Ready." / "What task should I run?"). Truncated to 10K chars with a
   direct-chat directive.
2. **Last-message pin** -- history trimming must never eat your actual question.
3. **Tool whitelist** -- opencode's 409-tool MCP catalog (~102K tokens) is
   filtered to core tools (bash, read, write, edit, glob, grep, ...) so the model
   keeps real power.
4. **No-tools honesty** -- when tools are dropped, the model is told explicitly,
   so it never announces actions it cannot take.

### Troubleshooting quick hits

| Symptom | Fix |
|---------|-----|
| "Ready." / "Context loaded." | Agent mode -- system truncation slipped; restart the proxy (start-muse-glimmer.ps1) |
| Empty answer after thinking | Reasoning consumed the budget; shorten the prompt |
| Raw `<\|message\|>` tokens | Chat template not loaded -- `--chat-template-file muse-template.jinja` |
| No response at all | Your prompt was trimmed away; start a fresh session |
| 1-3 tok/s | Something else holds the GPU -- stop the other engine |

## 7. Ecosystem & getting started

- **Download:** https://huggingface.co/meta-models/Muse-Glimmer-30B (weights) ·
  GGUFs: `meta-models/Muse-Glimmer-30B-GGUF` (kquant-17gb, kquant-dynamic) ·
  Unsloth dynamic quants: `unsloth/Muse-Glimmer-30B-GGUF`
- **Run it:** llama.cpp (what the fleet uses), MLX, ExecuTorch -- optimized
  integrations landed within days of release; vLLM/SGLang for scale; Ollama,
  LM Studio, Unsloth partnerships followed shortly after
- **Tune it:** TorchTitan (PyTorch) for further fine-tuning
- **Scaffolds:** OpenClaw and other agentic orchestration patterns
- **Dev docs:** https://dev.meta.ai/docs/muse-glimmer

## 8. Vs DeepSeek V4 Flash (our cloud workhorse)

The fleet's other daily driver is **DeepSeek V4 Flash via the DeepSeek cloud API**
(the model behind many opencode sessions). They are complementary, not rivals:

| Dimension | Muse Glimmer 30B (local) | DeepSeek V4 Flash (cloud) |
|-----------|--------------------------|---------------------------|
| **Cost** | Free -- your GPU, your electricity | API-metered, per-token. Cheap per token (DeepSeek's pricing model), but agentic loops burn a lot of tokens, so not free |
| **Privacy** | Fully local, zero telemetry, works offline | Prompts leave the machine (DeepSeek servers) |
| **VRAM** | ~21 GB of the 4090 while running | None -- zero GPU footprint, works alongside anything |
| **Multimodal** | Yes (images in) | No -- text only (a gap; DeepSeek should add it) |
| **Speed** | ~40 tok/s (DFlash drafter), but 60-90s load time when cold | Fast TTFT, no cold start, always available |
| **Context** | 131K local | Cloud-grade large context |
| **Reasoning** | Yes, controllable (low..xhigh) | Yes |
| **Availability** | Only when the server is running | 24/7, internet required |
| **Tools/agentic** | Strong (MCP-Atlas 75.5) | Strong (V4 line is agentic-trained) |

### Tandem or alternate? -- Tandem, by role

Because DS V4 Flash lives in the cloud, it costs **zero VRAM** -- there is no
reason to choose between them. Run them **together**, with clear roles:

- **Glimmer** = the local workhorse for deep agentic sessions, tool-heavy
  workflows, anything with images, privacy-sensitive work, and when you want
  zero marginal cost. Cost: it owns the 4090 for the duration.
- **DeepSeek V4 Flash** = the always-on lane: parallel second agent, quick
  questions while Glimmer is busy, cost-sensitive bulk work, and the fallback
  when the GPU is needed for something else (Ollama models, training).
- **Alternate (one at a time) only** applies to *local* engines fighting over
  VRAM (Glimmer vs Ollama) -- never between Glimmer and DS Flash.

Practical opencode pattern: one session on `local-llama/muse-glimmer-30b` for
the main agentic task, a second session on `deepseek/deepseek-v4-flash` for
parallel side work. Glimmer for the heavy lift, Flash for the steady hum.

## 9. Safety (summary)
Meta assessed Glimmer under its Advanced AI Scaling Framework; it is not a
"Frontier AI" model (weaker than Muse Spark). Designations: **Chem/Bio:
moderate or lower**, **Cyber: moderate or lower (inferred)**, **Loss of control:
moderate or lower (inferred)**. Train-time mitigations: safety SFT (tool-use
boundaries, prompt-injection resistance), safety RL, and appropriate-information-
flow training. Meta recommends deploying it inside a system with guardrails --
especially **human-in-the-loop confirmation for irreversible actions** when used
agentically. Not intended for users under 18; audio I/O unsupported.
