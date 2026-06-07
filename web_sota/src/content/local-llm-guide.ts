export type GuideTabId = "overview" | "hardware" | "providers" | "models" | "hub" | "fleet";

export interface GuideTab {
  id: GuideTabId;
  label: string;
  description: string;
}

export interface ExternalResource {
  name: string;
  url: string;
  blurb: string;
}

export interface ProviderGuide {
  id: string;
  name: string;
  homepage: string;
  docsUrl?: string;
  defaultPort: string;
  apiStyle: string;
  bestFor: string[];
  tradeoffs: string[];
  setupSteps: string[];
  localLlmMcpEnv: string;
  badge?: "recommended" | "advanced" | "catalog";
}

export interface HardwareTier {
  name: string;
  vram: string;
  ram: string;
  exampleModels: string;
  notes: string;
}

export const guideTabs: GuideTab[] = [
  {
    id: "overview",
    label: "Overview",
    description: "What local LLM means in this fleet and when to use the hub.",
  },
  {
    id: "hardware",
    label: "Hardware",
    description: "VRAM, RAM, and GPU tiers for realistic model choices.",
  },
  {
    id: "providers",
    label: "Providers",
    description: "Ollama, LM Studio, vLLM, and Hugging Face — compare and choose.",
  },
  {
    id: "models",
    label: "Models",
    description: "Sizes, formats, and where to find weights.",
  },
  {
    id: "hub",
    label: "This hub",
    description: "local-llm-mcp tools, ports, and dashboard.",
  },
  {
    id: "fleet",
    label: "Fleet",
    description: "How other MCP servers connect to local inference.",
  },
];

export const providerGuides: ProviderGuide[] = [
  {
    id: "ollama",
    name: "Ollama",
    homepage: "https://ollama.com",
    docsUrl: "https://github.com/ollama/ollama/blob/main/docs/api.md",
    defaultPort: "11434",
    apiStyle: "Native Ollama API + OpenAI-compatible `/v1`",
    badge: "recommended",
    bestFor: [
      "Fastest path to a working local model on Windows/macOS/Linux",
      "Pull-and-run workflow (`ollama pull`, `ollama run`)",
      "Default backend for fleet MCP sampling URLs",
      "Low-friction agentic workflows in peer MCP servers",
    ],
    tradeoffs: [
      "Less fine-grained serving control than vLLM",
      "Model library is curated — not every Hugging Face repo is one command away",
      "Throughput under heavy concurrent load is moderate vs dedicated servers",
    ],
    setupSteps: [
      "Install from ollama.com and start the daemon (tray app or `ollama serve`).",
      "Pull a model: `ollama pull llama3.2` or `ollama pull qwen2.5-coder:7b`.",
      "Verify: open http://127.0.0.1:11434 or run `ollama list`.",
      "Point this hub at `OLLAMA_BASE_URL=http://127.0.0.1:11434` in Settings.",
    ],
    localLlmMcpEnv: "OLLAMA_BASE_URL / PROVIDERS__OLLAMA_BASE_URL",
  },
  {
    id: "lmstudio",
    name: "LM Studio",
    homepage: "https://lmstudio.ai",
    docsUrl: "https://lmstudio.ai/docs",
    defaultPort: "1234",
    apiStyle: "OpenAI-compatible local server",
    badge: "recommended",
    bestFor: [
      "Visual model discovery, download, and load/unload",
      "Trying many GGUF builds without CLI friction",
      "Local OpenAI-compatible endpoint for experiments",
      "Side-by-side model comparison in a desktop UI",
    ],
    tradeoffs: [
      "Desktop app overhead — less ideal for headless servers",
      "API server must be started manually in the app",
      "Automation is weaker than Ollama CLI for fleet scripts",
    ],
    setupSteps: [
      "Install LM Studio and download a model from the in-app catalog.",
      "Start the local server (Developer tab) — default port 1234.",
      "Use base URL `http://127.0.0.1:1234/v1` for OpenAI-style clients.",
      "Set `LMSTUDIO_BASE_URL` in this hub's Settings.",
    ],
    localLlmMcpEnv: "LMSTUDIO_BASE_URL / PROVIDERS__LMSTUDIO_BASE_URL",
  },
  {
    id: "vllm",
    name: "vLLM",
    homepage: "https://docs.vllm.ai",
    docsUrl: "https://docs.vllm.ai/en/latest/getting_started/installation.html",
    defaultPort: "8000",
    apiStyle: "OpenAI-compatible high-throughput server",
    badge: "advanced",
    bestFor: [
      "Maximum tokens/sec on a dedicated NVIDIA GPU",
      "Continuous batching and multi-user concurrent inference",
      "Serving Hugging Face safetensors models at scale",
      "Production-like local or LAN inference",
    ],
    tradeoffs: [
      "Heavier install (CUDA, PyTorch stack); Windows support is limited",
      "You manage model paths, GPU memory, and server flags yourself",
      "Overkill for casual single-user chat",
    ],
    setupSteps: [
      "Install vLLM per official docs (Linux + NVIDIA is the happy path).",
      "Launch: `vllm serve <model> --port 8000` (or use repo docker-compose).",
      "Clients use `http://127.0.0.1:8000/v1`.",
      "Set `VLLM_BASE_URL` in Settings; use `llm_vllm` MCP tool when enabled.",
    ],
    localLlmMcpEnv: "VLLM_BASE_URL / PROVIDERS__VLLM_BASE_URL",
  },
  {
    id: "huggingface",
    name: "Hugging Face Hub",
    homepage: "https://huggingface.co",
    docsUrl: "https://huggingface.co/docs/hub/en/index",
    defaultPort: "—",
    apiStyle: "Model catalog + `transformers` / download APIs (not a runtime by itself)",
    badge: "catalog",
    bestFor: [
      "Finding model cards, licenses, and community quantizations",
      "Source weights for Ollama imports, LM Studio, or vLLM",
      "Researching benchmarks and model lineage",
      "Fine-tuning datasets and adapter repos",
    ],
    tradeoffs: [
      "Not an inference server — you still need Ollama, LM Studio, or vLLM to run models",
      "Large downloads; license varies per model",
      "GGUF vs safetensors choice affects which runtime you use",
    ],
    setupSteps: [
      "Search huggingface.co/models for task (chat, code, vision).",
      "Prefer models with clear license and active maintenance.",
      "GGUF files → LM Studio or Ollama import; safetensors → vLLM/transformers.",
      "Optional: `huggingface-cli login` for gated models.",
    ],
    localLlmMcpEnv: "Used via llm_huggingface tool + provider adapters (partial)",
  },
];

export const hardwareTiers: HardwareTier[] = [
  {
    name: "Minimal",
    vram: "8 GB",
    ram: "16 GB",
    exampleModels: "3B–7B at Q4 (e.g. Llama 3.2 3B, Phi-3 mini, Qwen2.5 7B)",
    notes: "CPU offload works but is slow. Stick to small models for interactive chat.",
  },
  {
    name: "Comfortable",
    vram: "12–16 GB",
    ram: "32 GB",
    exampleModels: "7B–14B Q4/Q5, small MoE (e.g. Qwen2.5 14B, Mistral 7B, Gemma 2 9B)",
    notes: "Sweet spot for daily coding assistants and fleet sampling backends.",
  },
  {
    name: "Enthusiast",
    vram: "24 GB",
    ram: "64 GB",
    exampleModels: "32B Q4, 70B Q2–Q3, larger MoE activations (Qwen 32B, DeepSeek distill)",
    notes: "Run one strong model at a time; watch context length vs VRAM.",
  },
  {
    name: "Workstation",
    vram: "32–48 GB+",
    ram: "64–128 GB",
    exampleModels: "70B Q4, multi-LoRA, vision models, vLLM concurrent users",
    notes: "vLLM and batch workloads become practical; keep cooling and power in mind.",
  },
];

export const modelLandscapeNotes = [
  {
    title: "Parameters vs VRAM",
    body: "Rule of thumb: Q4 quantization needs roughly 0.5–0.7 GB per billion parameters for weights alone. Add 2–8+ GB for long context and KV cache.",
  },
  {
    title: "GGUF vs safetensors",
    body: "GGUF is common for Ollama and LM Studio desktop loads. Safetensors is standard for vLLM and Hugging Face training/inference pipelines.",
  },
  {
    title: "MoE (Mixture of Experts)",
    body: "Models like Qwen-MoE advertise large total params but activate fewer per token — good speed if your runtime supports them well.",
  },
  {
    title: "Coding vs chat",
    body: "Coder-tuned weights (Qwen Coder, DeepSeek Coder, Codestral) outperform general chat models on repo tasks at the same size.",
  },
  {
    title: "Vision",
    body: "Multimodal models need extra VRAM. Use the Vision page in this dashboard when the loaded provider exposes image-capable weights.",
  },
];

export const fleetResources: ExternalResource[] = [
  {
    name: "mcp-central-docs — Local LLM integration",
    url: "https://github.com/sandraschi/mcp-central-docs/tree/main/integrations/local-llm",
    blurb: "Fleet-wide orchestration notes and port registry.",
  },
  {
    name: "local-llm-mcp README",
    url: "https://github.com/sandraschi/local-llm-mcp",
    blurb: "Install, MCP client snippet, and honest capability matrix.",
  },
  {
    name: "Ollama model library",
    url: "https://ollama.com/library",
    blurb: "Official pull targets for the default fleet sampling stack.",
  },
];

export const providerComparisonRows = [
  {
    dimension: "Ease of setup",
    ollama: "★★★★★",
    lmstudio: "★★★★☆",
    vllm: "★★☆☆☆",
    huggingface: "N/A (catalog)",
  },
  {
    dimension: "Throughput (single GPU)",
    ollama: "Good",
    lmstudio: "Good",
    vllm: "Excellent",
    huggingface: "—",
  },
  {
    dimension: "GUI management",
    ollama: "Tray + CLI",
    lmstudio: "Full desktop UI",
    vllm: "CLI / Docker",
    huggingface: "Web hub",
  },
  {
    dimension: "Fleet default",
    ollama: "Yes (`:11434/v1`)",
    lmstudio: "Optional",
    vllm: "Power users",
    huggingface: "Upstream source",
  },
  {
    dimension: "OpenAI-compatible API",
    ollama: "Yes (`/v1`)",
    lmstudio: "Yes",
    vllm: "Yes",
    huggingface: "Via runtimes",
  },
];
