import { BookOpen, Loader2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

interface SkillSummary {
  name: string;
  title: string;
  words: number;
}

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:10833";

export function Skills() {
  const [skills, setSkills] = useState<SkillSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [content, setContent] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(`${API_BASE}/api/v1/skills`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = (await r.json()) as { skills: SkillSummary[] };
      setSkills(data.skills ?? []);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load skills");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const openSkill = useCallback(async (name: string) => {
    setSelected(name);
    setContent("");
    try {
      const r = await fetch(`${API_BASE}/api/v1/skills/${name}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = (await r.json()) as { content: string };
      setContent(data.content ?? "");
    } catch {
      setContent("Failed to load skill content.");
    }
  }, []);

  return (
    <div className="space-y-8" data-testid="skills-page">
      <div className="flex flex-col gap-2">
        <h2 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
          <BookOpen className="h-8 w-8 text-emerald-500" />
          Skills
        </h2>
        <p className="text-slate-400 max-w-2xl">
          Server skills teach AI clients how to use this server. These SKILL.md files are the same
          content injected as the chat base prompt.
        </p>
      </div>

      {loading && (
        <div className="flex items-center gap-3 text-slate-400">
          <Loader2 className="h-5 w-5 animate-spin text-emerald-500" />
          Loading skills...
        </div>
      )}

      {!loading && error && (
        <div className="p-6 rounded-2xl border border-red-500/20 bg-red-500/[0.04] text-sm text-red-300">
          Failed to load skills: {error}
          <button
            type="button"
            onClick={() => void load()}
            className="ml-4 rounded-lg bg-red-500/10 px-3 py-1 text-red-200 hover:bg-red-500/20"
          >
            Retry
          </button>
        </div>
      )}

      {!loading && !error && skills.length === 0 && (
        <div className="p-8 rounded-3xl border border-dashed border-white/10 text-center text-slate-500">
          No skills found. Add a{" "}
          <code className="text-slate-300">skills/&lt;name&gt;/SKILL.md</code> to the server.
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-1 space-y-3">
          {skills.map((skill) => (
            <button
              key={skill.name}
              type="button"
              onClick={() => void openSkill(skill.name)}
              className={`w-full text-left rounded-xl border p-4 transition-all ${
                selected === skill.name
                  ? "border-emerald-500/50 bg-emerald-500/[0.06]"
                  : "border-white/10 bg-white/[0.02] hover:bg-white/[0.05]"
              }`}
              data-testid={`skill-${skill.name}`}
            >
              <div className="font-mono text-sm text-emerald-300">{skill.name}</div>
              {skill.title && (
                <div className="text-xs text-slate-500 mt-1 truncate">{skill.title}</div>
              )}
            </button>
          ))}
        </div>

        <div className="md:col-span-2">
          <Card className="border-white/5 bg-white/[0.02] backdrop-blur-xl border shadow-2xl">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg text-white">{selected ?? "Select a skill"}</CardTitle>
              <CardDescription className="text-slate-400 text-sm">SKILL.md content</CardDescription>
            </CardHeader>
            <CardContent>
              {!selected && (
                <div className="text-sm text-slate-500 py-8 text-center">
                  Choose a skill from the list to view its content.
                </div>
              )}
              {selected && content && (
                <pre className="whitespace-pre-wrap break-words rounded-xl bg-black/40 border border-white/5 p-4 text-xs text-slate-300 leading-relaxed max-h-[60vh] overflow-y-auto font-mono">
                  {content}
                </pre>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
