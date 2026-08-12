import {
  Activity,
  ArrowRight,
  Brain,
  CheckCircle2,
  Cpu,
  ExternalLink,
  FileText,
  Loader2,
  RefreshCw,
  Server,
  Sparkles,
  XCircle,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:10833";

interface EngineState {
  engine: string;
  running: boolean;
  server_port_open?: boolean;
  proxy_port_open?: boolean;
  processes: { pid: number; name: string }[];
  process_vram_gb?: Record<string, number>;
  gpu_vram?: Record<string, unknown>;
  loaded_models: string[];
  url?: string;
  server_url?: string;
}

interface EnginesResponse {
  engines: { llama?: EngineState; ollama?: EngineState };
  error?: string;
}

type EngineId = "llama" | "ollama";

const GLIMMER_FACTS: [string, string][] = [
  ["Parameters", "30B (Q4 quantized, fits 24 GB VRAM)"],
  ["Context", "131,072 tokens"],
  ["Modality", "Multimodal - text + images in, text out"],
  ["Speed", "~40 tok/s on RTX 4090 with DFlash drafter"],
  ["License", "Apache 2.0"],
  ["Reasoning", "low / medium / high / xhigh, extracted thoughts"],
];

function StatusPill({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium border ${
        ok
          ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
          : "border-red-500/30 bg-red-500/10 text-red-300"
      }`}
    >
      {ok ? <CheckCircle2 className="h-3.5 w-3.5" /> : <XCircle className="h-3.5 w-3.5" />}
      {label}
    </span>
  );
}

export function Glimmer() {
  const [engines, setEngines] = useState<EnginesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [restarting, setRestarting] = useState(false);
  const [restartMsg, setRestartMsg] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const r = await fetch(`${API_BASE}/api/v1/engines`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setEngines((await r.json()) as EnginesResponse);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load engine status");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
    const interval = setInterval(() => void load(), 10_000);
    return () => clearInterval(interval);
  }, [load]);

  const restart = useCallback(
    async (engine: EngineId) => {
      setRestarting(true);
      setRestartMsg(null);
      try {
        const r = await fetch(`${API_BASE}/api/v1/engines/${engine}/restart`, { method: "POST" });
        const data = (await r.json()) as { success: boolean; error?: string };
        if (data.success) {
          setRestartMsg(`${engine === "llama" ? "llama-server" : "Ollama"} restarted.`);
        } else {
          setRestartMsg(`Restart failed: ${data.error ?? "unknown error"}`);
        }
        await load();
      } catch (e) {
        setRestartMsg(`Restart failed: ${e instanceof Error ? e.message : String(e)}`);
      } finally {
        setRestarting(false);
      }
    },
    [load],
  );

  const startGlimmer = useCallback(async () => {
    setRestarting(true);
    setRestartMsg(null);
    try {
      const r = await fetch(`${API_BASE}/api/v1/engines/llama/start`, { method: "POST" });
      const data = (await r.json()) as {
        success: boolean;
        error?: string;
        evicted_ollama_models?: string[];
      };
      const evicted = (data.evicted_ollama_models ?? []).join(", ");
      if (data.success) {
        setRestartMsg(evicted ? `Started. Evicted Ollama tenants: ${evicted}.` : "Started.");
      } else {
        setRestartMsg(`Start failed: ${data.error ?? "unknown error"}`);
      }
      await load();
    } catch (e) {
      setRestartMsg(`Start failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setRestarting(false);
    }
  }, [load]);

  const llama = engines?.engines?.llama;
  const ollama = engines?.engines?.ollama;
  const vram = (llama?.gpu_vram ?? {}) as { used_gb?: number; total_gb?: number; free_gb?: number };
  const llamaVram = (llama?.process_vram_gb ?? {}) as Record<string, number>;

  return (
    <div className="space-y-8" data-testid="glimmer-page">
      <div className="flex flex-col gap-2">
        <h2 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
          <Sparkles className="h-8 w-8 text-emerald-500" />
          Muse Glimmer 30B
        </h2>
        <p className="text-slate-400 max-w-2xl">
          Meta's agentic multimodal model, served locally by llama.cpp (port 11439) behind the
          truncating proxy (port 11435). Live engine state below.
        </p>
      </div>

      {loading && (
        <div className="flex items-center gap-3 text-slate-400">
          <Loader2 className="h-5 w-5 animate-spin text-emerald-500" />
          Probing engines...
        </div>
      )}

      {!loading && error && (
        <div
          className="p-6 rounded-2xl border border-red-500/20 bg-red-500/[0.04] text-sm text-red-300"
          data-testid="glimmer-error"
        >
          Failed to load engine status: {error}
          <button
            type="button"
            onClick={() => void load()}
            className="ml-4 rounded-lg bg-red-500/10 px-3 py-1 text-red-200 hover:bg-red-500/20"
          >
            Retry
          </button>
        </div>
      )}

      {!loading && !error && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card
              className="border-white/5 bg-white/[0.02] backdrop-blur-xl border shadow-2xl"
              data-testid="engine-llama"
            >
              <CardHeader className="pb-2">
                <CardTitle className="text-base text-white flex items-center gap-2">
                  <Server className="h-4 w-4 text-emerald-500" />
                  llama-server
                </CardTitle>
                <CardDescription className="text-slate-400 text-sm">
                  Port 11439 - inference engine
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                <StatusPill
                  ok={!!llama?.server_port_open}
                  label={llama?.server_port_open ? "Listening" : "Down"}
                />
                <div className="text-xs font-mono text-slate-500">
                  {llama?.processes?.length
                    ? llama.processes.map((p) => `PID ${p.pid} (${p.name})`).join(", ")
                    : "no process"}
                </div>
                {llama?.loaded_models?.length ? (
                  <div className="text-xs text-slate-300">
                    Model:{" "}
                    <span className="font-mono text-emerald-300">
                      {llama.loaded_models.join(", ")}
                    </span>
                  </div>
                ) : (
                  <div className="text-xs text-slate-500">No model loaded</div>
                )}
              </CardContent>
            </Card>

            <Card
              className="border-white/5 bg-white/[0.02] backdrop-blur-xl border shadow-2xl"
              data-testid="engine-proxy"
            >
              <CardHeader className="pb-2">
                <CardTitle className="text-base text-white flex items-center gap-2">
                  <Activity className="h-4 w-4 text-amber-500" />
                  Truncating Proxy
                </CardTitle>
                <CardDescription className="text-slate-400 text-sm">
                  Port 11435 - front door
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                <StatusPill
                  ok={!!llama?.proxy_port_open}
                  label={llama?.proxy_port_open ? "Listening" : "Down"}
                />
                <div className="text-xs text-slate-500">
                  Trims oversized requests, pins your message, whitelists tools.
                </div>
              </CardContent>
            </Card>

            <Card
              className="border-white/5 bg-white/[0.02] backdrop-blur-xl border shadow-2xl"
              data-testid="engine-ollama"
            >
              <CardHeader className="pb-2">
                <CardTitle className="text-base text-white flex items-center gap-2">
                  <Cpu className="h-4 w-4 text-sky-500" />
                  Ollama
                </CardTitle>
                <CardDescription className="text-slate-400 text-sm">
                  Port 11434 - classic stack
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                <StatusPill
                  ok={!!ollama?.running}
                  label={ollama?.running ? "Running" : "Stopped"}
                />
                {ollama?.loaded_models?.length ? (
                  <div className="text-xs text-slate-300">
                    Loaded:{" "}
                    <span className="font-mono text-emerald-300">
                      {ollama.loaded_models.join(", ")}
                    </span>
                  </div>
                ) : (
                  <div className="text-xs text-slate-500">
                    No models loaded (Glimmer holds VRAM)
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card
              className="border-white/5 bg-white/[0.02] backdrop-blur-xl border shadow-2xl"
              data-testid="glimmer-gpu"
            >
              <CardHeader className="pb-2">
                <CardTitle className="text-base text-white flex items-center gap-2">
                  <Brain className="h-4 w-4 text-emerald-500" />
                  GPU State
                </CardTitle>
                <CardDescription className="text-slate-400 text-sm">
                  RTX 4090 - 24 GB VRAM
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                {vram.total_gb ? (
                  <>
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-400">Used</span>
                      <span className="text-white font-mono">{vram.used_gb?.toFixed(1)} GB</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-400">Free</span>
                      <span className="text-white font-mono">{vram.free_gb?.toFixed(1)} GB</span>
                    </div>
                    <div className="h-2 rounded-full bg-white/5 overflow-hidden">
                      <div
                        className="h-full bg-emerald-500/70 transition-all"
                        style={{
                          width: `${vram.total_gb ? Math.min(100, ((vram.used_gb ?? 0) / vram.total_gb) * 100) : 0}%`,
                        }}
                      />
                    </div>
                  </>
                ) : (
                  <div className="text-sm text-slate-500">
                    GPU probe returned no data (server down?).
                  </div>
                )}
                {Object.keys(llamaVram).length > 0 && (
                  <div className="text-xs text-slate-400">
                    llama-server VRAM:{" "}
                    {Object.entries(llamaVram)
                      .map(([pid, gb]) => `PID ${pid}: ${gb.toFixed(1)} GB`)
                      .join(", ")}
                  </div>
                )}
              </CardContent>
            </Card>

            <Card
              className="border-white/5 bg-white/[0.02] backdrop-blur-xl border shadow-2xl"
              data-testid="glimmer-facts"
            >
              <CardHeader className="pb-2">
                <CardTitle className="text-base text-white flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-emerald-500" />
                  Model Facts
                </CardTitle>
                <CardDescription className="text-slate-400 text-sm">
                  Muse Glimmer 30B (Meta, Apache 2.0)
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                {GLIMMER_FACTS.map(([k, v]) => (
                  <div key={k} className="flex justify-between gap-4 text-sm">
                    <span className="text-slate-400">{k}</span>
                    <span className="text-right text-white">{v}</span>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>

          <Card
            className="border-white/5 bg-white/[0.02] backdrop-blur-xl border shadow-2xl"
            data-testid="glimmer-controls"
          >
            <CardHeader className="pb-2">
              <CardTitle className="text-base text-white flex items-center gap-2">
                <RefreshCw className="h-4 w-4 text-emerald-500" />
                Engine Control
              </CardTitle>
              <CardDescription className="text-slate-400 text-sm">
                Starting Glimmer evicts any Ollama tenants (keep_alive=0) so the ~17 GB model fits
                the 4090, then loads llama-server (takes a few minutes). Restarting reloads the
                model the same way.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-wrap items-center gap-3">
              {llama?.server_port_open || llama?.proxy_port_open ? (
                <button
                  type="button"
                  onClick={() => void restart("llama")}
                  disabled={restarting}
                  className="rounded-xl bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-50 transition-colors"
                  data-testid="restart-llama"
                >
                  {restarting ? (
                    <span className="flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" /> Restarting...
                    </span>
                  ) : (
                    "Restart llama-server"
                  )}
                </button>
              ) : (
                <button
                  type="button"
                  onClick={() => void startGlimmer()}
                  disabled={restarting}
                  className="rounded-xl bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-50 transition-colors"
                  data-testid="start-glimmer"
                >
                  {restarting ? (
                    <span className="flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" /> Starting...
                    </span>
                  ) : ollama?.running ? (
                    "Start Glimmer (evicts Ollama)"
                  ) : (
                    "Start Glimmer"
                  )}
                </button>
              )}
              <button
                type="button"
                onClick={() => void restart("ollama")}
                disabled={restarting}
                className="rounded-xl bg-white/5 border border-white/10 px-4 py-2 text-sm font-medium text-slate-200 hover:bg-white/10 disabled:opacity-50 transition-colors"
                data-testid="restart-ollama"
              >
                Restart Ollama
              </button>
              {restartMsg && <span className="text-sm text-slate-300">{restartMsg}</span>}
            </CardContent>
          </Card>

          <div className="flex flex-wrap gap-4 text-sm">
            <a
              href="http://127.0.0.1:10832/help"
              className="inline-flex items-center gap-2 text-emerald-400 hover:text-emerald-300"
            >
              <FileText className="h-4 w-4" /> Glimmer tab in Help{" "}
              <ArrowRight className="h-4 w-4" />
            </a>
            <a
              href="https://github.com/sandraschi/local-llm-mcp/blob/main/docs/GLIMMER.md"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 text-slate-400 hover:text-slate-200"
            >
              <ExternalLink className="h-4 w-4" /> docs/GLIMMER.md
            </a>
          </div>
        </>
      )}
    </div>
  );
}
