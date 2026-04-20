# Qwen 3.6-35B-A3B: The Agentic Coding Standard

Released on April 15, 2026, **Qwen 3.6-35B-A3B** is the flagship Mixture-of-Experts (MoE) model for the local Agentic Fleet. It is specifically optimized for repository-level reasoning, complex tool-calling, and front-end workflow automation.

## 🏗️ Architecture: A3B (Activated 3 Billion)

Qwen 3.6-35B-A3B uses a **Sparse MoE** architecture. While the model has 35 billion parameters (providing a massive knowledge base), only **3 billion parameters** are activated for each token.

### Key Benefits:
- **Large-Model Reasoning**: Maintains the logic and zero-shot capabilities of 30B+ parameter models.
- **Small-Model Speed**: Achieves inference latencies comparable to dense 3B-7B models.
- **Local Optimization**: Perfectly fits within 24GB VRAM (FP16) or 12GB VRAM (Q4_K_M) while outperforming much larger dense models.

## 🚀 SOTA Integration Guide

### 1. Automatic Discovery (Preferred)
The `local-llm-mcp` server follows the **SOTA v14.1** industrial standard for automatic model discovery.

1.  **Ollama**: Run `ollama run qwen3.6:35b-a3b`.
2.  **Dashboard**: The [Orchestration Dashboard](http://localhost:10832) will detect the new model automatically on heartbeat.
3.  **Defaulting**: Once detected, the fleet will prioritize this model for any `agentic-coding` prompts.

### 2. Manual Configuration (Advanced)
If you are running Qwen 3.6 via a custom vLLM endpoint or LM Studio with a specific port:

Update your `config.yaml`:
```yaml
model:
  default_model: "qwen3.6:35b-a3b"
  recommended_models:
    - "qwen3.6:35b-a3b"
    - "microsoft/Phi-4" # Legacy
```

## 🧠 Agentic Benchmarks (April 2026)

| Benchmark | Qwen 3.6-35B-A3B | Google Gemma 4 (7B) | Llama 3.1 (8B) |
|-----------|------------------|----------------------|----------------|
| **HumanEval** | **89.4%** | 82.1% | 74.5% |
| **MBPP** | **86.2%** | 80.5% | 72.1% |
| **Agentic Tool-Use** | **94.1%** | 88.3% | 81.0% |
| **MoE Efficiency** | **11.2x** | N/A (Dense) | N/A (Dense) |

---
> [!TIP]
> **Pro-Tip**: Use the `qwen3.6:35b-a3b-fp16` tag if you have 32GB+ VRAM for maximum precision in complex architectural refactorings.
