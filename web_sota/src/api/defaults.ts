const KEY = "llm-mcp-defaults";

export interface Defaults {
  provider: string;
  model: string;
}

export function getDefaults(): Defaults | null {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const d = JSON.parse(raw) as Defaults;
    return d && typeof d.provider === "string" && typeof d.model === "string" ? d : null;
  } catch {
    return null;
  }
}

export function setDefaults(provider: string, model: string): void {
  localStorage.setItem(KEY, JSON.stringify({ provider, model }));
}
