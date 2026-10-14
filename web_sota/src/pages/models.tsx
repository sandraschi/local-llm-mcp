import { Brain, Loader2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
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
