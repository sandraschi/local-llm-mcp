const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:10833";

export async function getHealth(): Promise<{ status: string }> {
  const r = await fetch(`${API_BASE}/health`);
  if (!r.ok) throw new Error(`Health check failed: ${r.status}`);
  return r.json();
}

export interface Config {
  providers: {
    ollama_base_url: string;
    vllm_base_url: string;
    lmstudio_base_url: string;
    openai_api_key?: string;
    gemini_api_key?: string;
  };
  server: {
    port: number;
    log_level: string;
  };
}

export async function getConfig(): Promise<Config> {
  const res = await fetch(`${API_BASE}/config`);
  if (!res.ok) throw new Error("Failed to fetch config");
  return res.json();
}

export async function updateConfig(updates: Partial<Config>): Promise<Config> {
  const res = await fetch(`${API_BASE}/config`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(updates),
  });
  if (!res.ok) throw new Error("Failed to update config");
  return res.json();
}

export interface ModelIntelligence {
  hf_id?: string;
  developer?: string;
  release_date?: string;
  strengths: string[];
  weaknesses: string[];
  best_for?: string;
  is_legacy: boolean;
  vram_required_gb?: number;
  quantization_info?: string;
  model_card_url?: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  provider: string;
  description?: string;
  capabilities?: string[];
  intelligence?: ModelIntelligence;
  hardware_compatibility?: "READY" | "TIGHT" | "OOM" | "UNKNOWN";
}

export async function listModels(): Promise<ModelInfo[]> {
  const r = await fetch(`${API_BASE}/api/v1/models`);
  if (!r.ok) throw new Error(`Models list failed: ${r.status}`);
  return r.json();
}

export interface GenerateResponse {
  text: string;
  model: string;
  provider?: string;
  usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number };
}

export async function generate(
  prompt: string,
  model: string,
  provider?: string,
): Promise<GenerateResponse> {
  const r = await fetch(`${API_BASE}/api/v1/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, model, provider, stream: false, max_tokens: 512 }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? `Generate failed: ${r.status}`);
  }
  return r.json();
}
