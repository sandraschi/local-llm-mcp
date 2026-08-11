import { Loader2, Search, Wrench } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

interface ToolInfo {
  name: string;
  description: string;
}

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:10833";

export function Tools() {
  const [tools, setTools] = useState<ToolInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(`${API_BASE}/api/v1/tools`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = (await r.json()) as { tools: ToolInfo[]; total: number };
      setTools(data.tools ?? []);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load tools");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const filtered = query.trim()
    ? tools.filter(
        (t) =>
          t.name.toLowerCase().includes(query.toLowerCase()) ||
          (t.description ?? "").toLowerCase().includes(query.toLowerCase()),
      )
    : tools;

  return (
    <div className="space-y-8" data-testid="tools-page">
      <div className="flex flex-col gap-2">
        <h2 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
          <Wrench className="h-8 w-8 text-emerald-500" />
          MCP Tools
        </h2>
        <p className="text-slate-400 max-w-2xl">
          Live catalog of every tool registered by the MCP server. Each portmanteau tool
          consolidates related operations behind an operation enum.
        </p>
      </div>

      <div className="relative max-w-md">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search tools..."
          className="w-full rounded-xl border border-white/10 bg-white/[0.03] py-2.5 pl-10 pr-4 text-sm text-white placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500/40"
          data-testid="tools-search"
        />
      </div>

      {loading && (
        <div className="flex items-center gap-3 text-slate-400" data-testid="tools-loading">
          <Loader2 className="h-5 w-5 animate-spin text-emerald-500" />
          Loading tool catalog (first load initializes the MCP server)...
        </div>
      )}

      {!loading && error && (
        <div
          className="p-6 rounded-2xl border border-red-500/20 bg-red-500/[0.04] text-sm text-red-300"
          data-testid="tools-error"
        >
          Failed to load tools: {error}
          <button
            type="button"
            onClick={() => void load()}
            className="ml-4 rounded-lg bg-red-500/10 px-3 py-1 text-red-200 hover:bg-red-500/20"
          >
            Retry
          </button>
        </div>
      )}

      {!loading && !error && tools.length === 0 && (
        <div
          className="p-8 rounded-3xl border border-dashed border-white/10 text-center text-slate-500"
          data-testid="tools-empty"
        >
          No tools registered. Start the MCP server and retry.
        </div>
      )}

      {!loading && !error && filtered.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {filtered.map((tool) => (
            <Card
              key={tool.name}
              className="border-white/5 bg-white/[0.02] hover:bg-white/[0.05] backdrop-blur-xl transition-all border shadow-2xl overflow-hidden hover:border-emerald-500/40"
            >
              <CardHeader className="pb-2">
                <CardTitle className="text-base font-mono text-emerald-300">{tool.name}</CardTitle>
                <CardDescription className="text-slate-400 text-sm">
                  {tool.description || "No description"}
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-0">
                <span className="inline-flex px-2 py-0.5 rounded-full text-[10px] uppercase tracking-wider font-bold bg-white/5 text-slate-500 border border-white/5">
                  MCP tool
                </span>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {!loading && !error && filtered.length === 0 && tools.length > 0 && (
        <div className="p-8 rounded-3xl border border-dashed border-white/10 text-center text-slate-500">
          No tools match "{query}".
        </div>
      )}
    </div>
  );
}
