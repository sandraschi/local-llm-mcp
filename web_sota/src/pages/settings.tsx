import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { listModels, type ModelInfo } from "@/api/client";
import { getDefaults, setDefaults } from "@/api/defaults";

function uniqueProviders(models: ModelInfo[]): string[] {
  const set = new Set(models.map((m) => m.provider).filter(Boolean));
  return Array.from(set).sort();
}

export function Settings() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");
  const [saved, setSaved] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const providers = uniqueProviders(models);
  const modelsForProvider = provider
    ? models.filter((m) => m.provider === provider)
    : models;

  useEffect(() => {
    listModels()
      .then((list) => {
        setModels(list);
        const d = getDefaults();
        if (d) {
          setProvider(d.provider);
          if (list.some((m) => m.provider === d.provider && (m.id === d.model || m.name === d.model))) {
            setModel(d.model);
          } else if (list.length > 0) {
            setModel(list[0].id ?? list[0].name);
          }
        } else if (list.length > 0) {
          const first = list[0];
          setProvider(first.provider);
          setModel(first.id ?? first.name);
        }
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load models"))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (provider && modelsForProvider.length > 0 && !modelsForProvider.some((m) => (m.id ?? m.name) === model)) {
      setModel(modelsForProvider[0].id ?? modelsForProvider[0].name);
    }
  }, [provider, modelsForProvider, model]);

  function handleSave() {
    if (provider && model) {
      setDefaults(provider, model);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    }
  }

  if (loading) {
    return (
      <div className="space-y-6">
        <h2 className="text-2xl font-bold tracking-tight text-white">Settings</h2>
        <p className="text-slate-500">Loading providers and models…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-6">
        <h2 className="text-2xl font-bold tracking-tight text-white">Settings</h2>
        <p className="text-red-400">{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight text-white">Settings</h2>
        <p className="text-slate-400 mt-1">Default provider and model for Chat. Stored in this browser.</p>
      </div>

      <Card className="border-slate-800 bg-slate-950/50 max-w-md">
        <CardHeader>
          <CardTitle className="text-white">Default provider & model</CardTitle>
          <CardDescription className="text-slate-400">Used when you open Chat</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-2">
            <Label className="text-slate-300">Provider</Label>
            <select
              value={provider}
              onChange={(e) => setProvider(e.target.value)}
              className="bg-slate-900 border border-slate-700 rounded-md px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-emerald-500"
            >
              {providers.length === 0 && <option value="">No providers</option>}
              {providers.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </div>
          <div className="grid gap-2">
            <Label className="text-slate-300">Model</Label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="bg-slate-900 border border-slate-700 rounded-md px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-emerald-500"
            >
              {modelsForProvider.length === 0 && <option value="">No models</option>}
              {modelsForProvider.map((m) => {
                const id = m.id ?? m.name;
                return (
                  <option key={id} value={id}>{m.name}</option>
                );
              })}
            </select>
          </div>
          <Button
            onClick={handleSave}
            disabled={!provider || !model || saved}
            className="bg-emerald-600 hover:bg-emerald-700"
          >
            {saved ? "Saved" : "Save defaults"}
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
