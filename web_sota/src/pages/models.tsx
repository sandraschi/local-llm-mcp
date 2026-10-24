import {
  Brain,
  CheckCircle2,
  Download,
  FlaskConical,
  Loader2,
  RefreshCw,
  Trash2,
  XCircle,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

interface ModelCardSummary {
  name: string;
  title: string;
  status: string;
  last_updated: string;
  status_line: string;
  verdict: string;
  words: number;
}

interface ModelCardDetail extends ModelCardSummary {
  content: string;
}

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:10833";

function stripFrontmatter(md: string): string {
  if (md.startsWith("---")) {
    const end = md.indexOf("---", 3);
    if (end !== -1) return md.slice(end + 3).trimStart();
  }
  return md;
}

export function Models() {
  const [cards, setCards] = useState<ModelCardSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<ModelCardDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(`${API_BASE}/api/v1/model-cards`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = (await r.json()) as { cards: ModelCardSummary[]; error?: string };
      if (data.error) throw new Error(data.error);
      setCards(data.cards ?? []);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load model cards");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const openCard = useCallback(async (name: string) => {
    setSelected(name);
    setDetail(null);
    try {
      const r = await fetch(`${API_BASE}/api/v1/model-cards/${name}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = (await r.json()) as ModelCardDetail;
      setDetail(data);
    } catch {
      setDetail(null);
    }
  }, []);

  return (
    <div className="space-y-8" data-testid="models-page">
      <div className="flex flex-col gap-2">
        <h2 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
          <Brain className="h-8 w-8 text-emerald-500" />
          Models
        </h2>
        <p className="text-slate-400 max-w-2xl">
          Fleet model intel from the mcd info depot (<code className="text-slate-300">mcp-central-docs/models/</code>).
          What runs on the 4090, what needs a datacenter, and what to wait for.
        </p>
      </div>

      <OllamaLibrary />

      {loading && (
        <div className="flex items-center gap-3 text-slate-400">
          <Loader2 className="h-5 w-5 animate-spin text-emerald-500" />
          Loading model cards...
        </div>
      )}

      {!loading && error && (
        <div className="p-6 rounded-2xl border border-red-500/20 bg-red-500/[0.04] text-sm text-red-300">
          Failed to load model cards: {error}
          <button
            type="button"
            onClick={() => void load()}
            className="ml-4 rounded-lg bg-red-500/10 px-3 py-1 text-red-200 hover:bg-red-500/20"
          >
            Retry
          </button>
        </div>
      )}

      {!loading && !error && cards.length === 0 && (
        <div className="p-8 rounded-3xl border border-dashed border-white/10 text-center text-slate-500">
          No model cards found in the depot.
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-1 space-y-3">
          {cards.map((card) => (
            <button
              key={card.name}
              type="button"
              onClick={() => void openCard(card.name)}
              className={`w-full text-left rounded-xl border p-4 transition-all ${
                selected === card.name
                  ? "border-emerald-500/50 bg-emerald-500/[0.06]"
                  : "border-white/10 bg-white/[0.02] hover:bg-white/[0.05]"
              }`}
              data-testid={`model-card-${card.name}`}
            >
              <div className="font-mono text-sm text-emerald-300">{card.name}</div>
              {card.title && <div className="text-xs text-slate-400 mt-1">{card.title}</div>}
              {card.verdict && (
                <div className="text-xs text-slate-500 mt-1 truncate">{card.verdict}</div>
              )}
            </button>
          ))}
        </div>

        <div className="md:col-span-2">
          <Card className="border-white/5 bg-white/[0.02] backdrop-blur-xl border shadow-2xl">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg text-white">
                {detail?.title ?? selected ?? "Select a model"}
              </CardTitle>
              <CardDescription className="text-slate-400 text-sm">
                {detail?.status_line || "Model card from the info depot"}
                {detail?.last_updated && ` - updated ${detail.last_updated}`}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!selected && (
                <div className="text-sm text-slate-500 py-8 text-center">
                  Choose a model from the list to view its card.
                </div>
              )}
              {selected && !detail && (
                <div className="flex items-center gap-3 text-slate-400 py-8 justify-center">
                  <Loader2 className="h-5 w-5 animate-spin text-emerald-500" />
                  Loading card...
                </div>
              )}
              {selected && detail && (
                <div className="text-sm text-slate-300 max-h-[60vh] overflow-y-auto pr-2 prose prose-invert prose-sm max-w-none prose-p:my-2 prose-ul:my-2 prose-li:my-0 prose-table:text-xs prose-headings:text-slate-100">
                  <ReactMarkdown>{stripFrontmatter(detail.content)}</ReactMarkdown>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

interface InstalledModel {
  name: string;
  size_gb: number;
  modified_at: string;
  digest: string;
}

type TestState = { state: "idle" } | { state: "running" } | { state: "done"; ok: boolean; detail: string };

/** Ollama library: installed models with pull / test / remove. */
function OllamaLibrary() {
  const [lib, setLib] = useState<InstalledModel[]>([]);
  const [libLoading, setLibLoading] = useState(true);
  const [libError, setLibError] = useState<string | null>(null);
  const [pullName, setPullName] = useState("");
  const [pullJob, setPullJob] = useState<{ job_id: string; progress: number; status_text: string } | null>(null);
  const [tests, setTests] = useState<Record<string, TestState>>({});
  const pollRef = useRef<number | null>(null);

  const loadLib = useCallback(async () => {
    setLibLoading(true);
    setLibError(null);
    try {
      const r = await fetch(`${API_BASE}/api/v1/ollama/models`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = (await r.json()) as { models?: InstalledModel[]; error?: string };
      if (d.error) throw new Error(d.error);
      setLib(d.models ?? []);
    } catch (e) {
      setLibError(e instanceof Error ? e.message : "Failed to load Ollama library");
    } finally {
      setLibLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadLib();
  }, [loadLib]);

  useEffect(
    () => () => {
      if (pollRef.current !== null) window.clearInterval(pollRef.current);
    },
    [],
  );

  const pollPull = useCallback(
    (jobId: string) => {
      if (pollRef.current !== null) window.clearInterval(pollRef.current);
      pollRef.current = window.setInterval(async () => {
        try {
          const r = await fetch(`${API_BASE}/api/v1/ollama/pull/${jobId}`);
          const d = (await r.json()) as {
            state?: string;
            progress?: number;
            status_text?: string;
            error?: string;
          };
          if (d.state === "done" || d.state === "failed") {
            if (pollRef.current !== null) window.clearInterval(pollRef.current);
            setPullJob(null);
            if (d.state === "failed") setLibError(d.error || "Pull failed");
            else void loadLib();
          } else {
            setPullJob({ job_id: jobId, progress: d.progress ?? 0, status_text: d.status_text ?? "" });
          }
        } catch {
          /* keep polling */
        }
      }, 2000);
    },
    [loadLib],
  );

  const startPull = async () => {
    const name = pullName.trim();
    if (!name) return;
    setLibError(null);
    try {
      const r = await fetch(`${API_BASE}/api/v1/ollama/pull`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      const d = (await r.json()) as { success?: boolean; job_id?: string; error?: string };
      if (!d.success || !d.job_id) throw new Error(d.error || "Pull did not start");
      setPullName("");
      pollPull(d.job_id);
    } catch (e) {
      setLibError(e instanceof Error ? e.message : "Pull failed to start");
    }
  };

  const removeModel = async (name: string) => {
    if (!window.confirm(`Remove ${name} from Ollama (frees disk immediately)?`)) return;
    setLibError(null);
    try {
      const r = await fetch(`${API_BASE}/api/v1/ollama/delete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      const d = (await r.json()) as { success?: boolean; error?: string };
      if (!d.success) throw new Error(d.error || "Delete failed");
      await loadLib();
    } catch (e) {
      setLibError(e instanceof Error ? e.message : "Delete failed");
    }
  };

  const testModel = async (name: string) => {
    setTests((t) => ({ ...t, [name]: { state: "running" } }));
    try {
      const r = await fetch(`${API_BASE}/api/v1/ollama/test`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      const d = (await r.json()) as { ok?: boolean; seconds?: number; reply?: string; error?: string };
      setTests((t) => ({
        ...t,
        [name]: d.ok
          ? { state: "done", ok: true, detail: `${d.seconds ?? "?"}s - ${d.reply ?? ""}`.trim() }
          : { state: "done", ok: false, detail: d.error ?? "failed" },
      }));
    } catch (e) {
      setTests((t) => ({
        ...t,
        [name]: { state: "done", ok: false, detail: e instanceof Error ? e.message : "failed" },
      }));
    }
  };

  const totalGb = lib.reduce((s, m) => s + (m.size_gb || 0), 0);

  return (
    <Card className="border-white/5 bg-white/[0.02] backdrop-blur-xl shadow-2xl" data-testid="ollama-library">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg text-white">Ollama library</CardTitle>
        <CardDescription className="text-slate-400 text-sm">
          {lib.length} models installed ({totalGb.toFixed(1)} GB). Test runs a cold-load smoke check;
          Remove frees disk immediately. LM Studio models are managed in LM Studio itself.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-2" data-testid="ollama-pull-box">
          <input
            value={pullName}
            onChange={(e) => setPullName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void startPull();
            }}
            placeholder="ollama library name, e.g. qwen3:8b"
            className="flex-1 rounded-lg border border-white/10 bg-white/[0.03] px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:border-emerald-500/50 focus:outline-none"
            data-testid="ollama-pull-input"
          />
          <button
            type="button"
            onClick={() => void startPull()}
            className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500"
            data-testid="ollama-pull-start"
          >
            <span className="inline-flex items-center gap-1.5">
              <Download className="h-4 w-4" /> Pull
            </span>
          </button>
          <button
            type="button"
            onClick={() => void loadLib()}
            title="Refresh list"
            className="rounded-lg border border-white/10 px-3 py-2 text-slate-300 hover:bg-white/5"
            data-testid="ollama-refresh"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
        </div>

        {pullJob && (
          <div className="text-sm text-slate-300" data-testid="ollama-pull-progress">
            Pulling <span className="font-mono">{pullJob.status_text || "…"}</span>
            <div className="mt-1 h-1.5 rounded-full bg-white/10">
              <div
                className="h-1.5 rounded-full bg-emerald-500 transition-all"
                style={{ width: `${Math.round(pullJob.progress * 100)}%` }}
              />
            </div>
          </div>
        )}

        {libLoading && (
          <div className="flex items-center gap-3 text-slate-400 text-sm">
            <Loader2 className="h-4 w-4 animate-spin text-emerald-500" /> Loading library...
          </div>
        )}

        {libError && <div className="text-sm text-red-300">{libError}</div>}

        {!libLoading && !libError && lib.length > 0 && (
          <ul className="divide-y divide-white/5" data-testid="ollama-model-list">
            {lib.map((m) => {
              const t = tests[m.name] ?? { state: "idle" as const };
              return (
                <li key={m.name} className="flex items-center gap-3 py-2">
                  <div className="min-w-0 flex-1">
                    <div className="font-mono text-sm text-slate-200 truncate">{m.name}</div>
                    <div className="text-xs text-slate-500">
                      {m.size_gb} GB
                      {t.state === "done" && (
                        <span className={t.ok ? "text-emerald-400" : "text-red-400"}>
                          {" "}
                          -                           {t.ok ? <CheckCircle2 className="inline h-3 w-3" /> : <XCircle className="inline h-3 w-3" />}{" "}
                          {t.detail}
                        </span>
                      )}
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => void testModel(m.name)}
                    disabled={t.state === "running"}
                    title="Cold-load + tiny completion smoke test"
                    className="rounded-lg border border-white/10 px-2.5 py-1.5 text-xs text-slate-300 hover:bg-white/5 disabled:opacity-50"
                    data-testid={`ollama-test-${m.name}`}
                  >
                    {t.state === "running" ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    ) : (
                      <FlaskConical className="h-3.5 w-3.5" />
                    )}
                  </button>
                  <button
                    type="button"
                    onClick={() => void removeModel(m.name)}
                    title="Remove from Ollama"
                    className="rounded-lg border border-red-500/20 px-2.5 py-1.5 text-xs text-red-300 hover:bg-red-500/10"
                    data-testid={`ollama-delete-${m.name}`}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
