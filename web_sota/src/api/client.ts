const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:10833";

export async function getHealth(): Promise<{ status: string }> {
  const r = await fetch(`${API_BASE}/health`);
  if (!r.ok) throw new Error(`Health check failed: ${r.status}`);
  return r.json();
}

export interface ModelInfo {
  id: string;
  name: string;
  provider: string;
  description?: string;
  capabilities?: string[];
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

export async function generate(prompt: string, model: string, provider?: string): Promise<GenerateResponse> {
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
