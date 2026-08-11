import { CheckCircle2, Loader2, XCircle } from "lucide-react";
import { useEffect, useState } from "react";
import { Label } from "@/components/ui/label";

interface ProviderDef {
  id: string;
  label: string;
  port: number;
  modelsUrl: string;
  listModels: (data: unknown) => string[];
}

const PROVIDERS: ProviderDef[] = [
  {
    id: "ollama",
    label: "Ollama",
    port: 11434,
    modelsUrl: "http://localhost:11434/api/tags",
    listModels: (d) =>
      ((d as { models?: { name?: string }[] }).models ?? [])
        .map((m) => m.name ?? "")
        .filter(Boolean),
  },
  {
    id: "lmstudio",
    label: "LM Studio",
    port: 1234,
    modelsUrl: "http://localhost:1234/v1/models",
    listModels: (d) =>
      ((d as { data?: { id?: string }[] }).data ?? []).map((m) => m.id ?? "").filter(Boolean),
  },
  {
    id: "vllm",
    label: "vLLM",
    port: 8000,
    modelsUrl: "http://localhost:8000/v1/models",
    listModels: (d) =>
      ((d as { data?: { id?: string }[] }).data ?? []).map((m) => m.id ?? "").filter(Boolean),
  },
];

const PROVIDER_KEY = "llm_provider";
const MODEL_KEY = "llm_model";

type Status = "probing" | "detected" | "not_found";

async function probe(provider: ProviderDef, timeoutMs = 3000): Promise<Status> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const r = await fetch(provider.modelsUrl, { signal: controller.signal });
    return r.ok ? "detected" : "not_found";
  } catch {
    return "not_found";
  } finally {
    clearTimeout(timer);
  }
}

async function fetchModels(provider: ProviderDef): Promise<string[]> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 3000);
  try {
    const r = await fetch(provider.modelsUrl, { signal: controller.signal });
    if (!r.ok) return [];
    return provider.listModels(await r.json());
  } catch {
    return [];
  } finally {
    clearTimeout(timer);
  }
}

export function LocalEngineDetection() {
  const [statuses, setStatuses] = useState<Record<string, Status>>({});
  const [detected, setDetected] = useState<ProviderDef[]>([]);
  const [providerId, setProviderId] = useState<string>("");
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState<string>("");
  const [noGpuHint, setNoGpuHint] = useState(false);

  // (a) Auto-detect providers on mount
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const initial: Record<string, Status> = {};
      for (const p of PROVIDERS) initial[p.id] = "probing";
      setStatuses(initial);

      const results = await Promise.all(
        PROVIDERS.map(async (p) => ({ id: p.id, status: await probe(p) })),
      );
      if (cancelled) return;

      const next: Record<string, Status> = {};
      for (const r of results) next[r.id] = r.status;
      setStatuses(next);
      const found = PROVIDERS.filter((p) => next[p.id] === "detected");
      setDetected(found);

      // (g) Restore saved provider, else first detected
      const savedProvider = localStorage.getItem(PROVIDER_KEY) ?? "";
      const chosen = found.find((p) => p.id === savedProvider) ?? found[0] ?? null;
      if (chosen) {
        setProviderId(chosen.id);
        const savedModel = localStorage.getItem(MODEL_KEY) ?? "";
        const m = await fetchModels(chosen);
        if (cancelled) return;
        setModels(m);
        setModel(m.includes(savedModel) ? savedModel : (m[0] ?? ""));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // (f) Re-fetch models on provider change
  useEffect(() => {
    if (!providerId) return;
    const provider = PROVIDERS.find((p) => p.id === providerId);
    if (!provider) return;
    localStorage.setItem(PROVIDER_KEY, providerId);
    (async () => {
      const m = await fetchModels(provider);
      setModels(m);
      const savedModel = localStorage.getItem(MODEL_KEY) ?? "";
      setModel(m.includes(savedModel) ? savedModel : (m[0] ?? ""));
    })();
  }, [providerId]);

  // (e) Persist model selection
  useEffect(() => {
    if (model) localStorage.setItem(MODEL_KEY, model);
  }, [model]);

  // (i) GPU opportunity hint when no provider detected
  useEffect(() => {
    const anyDetected = Object.values(statuses).some((s) => s === "detected");
    const allDone = Object.values(statuses).every((s) => s !== "probing");
    if (allDone && !anyDetected) {
      const t = setTimeout(() => setNoGpuHint(true), 800);
      return () => clearTimeout(t);
    }
    setNoGpuHint(false);
    return undefined;
  }, [statuses]);

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {PROVIDERS.map((p) => {
          const status = statuses[p.id] ?? "probing";
          return (
            <div
              key={p.id}
              className="rounded-xl border border-white/10 bg-white/[0.03] p-3 flex items-center justify-between"
              data-testid={`provider-${p.id}`}
            >
              <div>
                <div className="text-sm font-medium text-white">{p.label}</div>
                <div className="text-[10px] font-mono text-slate-500">:{p.port}</div>
              </div>
              <div className="flex items-center gap-1.5 text-xs">
                {status === "probing" && (
                  <>
                    <Loader2 className="h-3.5 w-3.5 animate-spin text-amber-400" />
                    <span className="text-amber-300">Probing...</span>
                  </>
                )}
                {status === "detected" && (
                  <>
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
                    <span className="text-emerald-300">Detected</span>
                  </>
                )}
                {status === "not_found" && (
                  <>
                    <XCircle className="h-3.5 w-3.5 text-slate-600" />
                    <span className="text-slate-500">Not found</span>
                  </>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {detected.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <div className="space-y-2">
            <Label className="text-slate-300">Provider</Label>
            <select
              value={providerId}
              onChange={(e) => setProviderId(e.target.value)}
              data-testid="llm-provider-select"
              className="w-full rounded-xl border border-white/10 bg-zinc-800 text-zinc-100 px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/40"
            >
              {detected.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
            </select>
          </div>
          <div className="space-y-2">
            <Label className="text-slate-300">Model</Label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              data-testid="llm-model-select"
              className="w-full rounded-xl border border-white/10 bg-zinc-800 text-zinc-100 px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/40"
            >
              {models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
        </div>
      ) : (
        <div className="rounded-xl border border-dashed border-white/10 bg-white/[0.02] p-4 text-sm text-slate-400">
          {Object.values(statuses).some((s) => s === "probing")
            ? "Probing local LLM engines..."
            : "No local LLM engine detected. Start Ollama, LM Studio, or vLLM to enable AI features."}
        </div>
      )}

      {noGpuHint && (
        <div
          className="rounded-xl border border-amber-500/20 bg-amber-500/[0.05] p-4 text-sm text-amber-200"
          data-testid="llm-gpu-hint"
        >
          High-performance GPU detected. Install Ollama or LM Studio to unlock free local AI
          features.
        </div>
      )}
    </div>
  );
}
