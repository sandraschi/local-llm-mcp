import { Check, Cpu, Globe, Loader2, Save, Server, Settings as SettingsIcon } from "lucide-react";
import { useEffect, useState } from "react";
import { getConfig, listModels, type ModelInfo, updateConfig } from "@/api/client";
import { getDefaults, setDefaults } from "@/api/defaults";
import { cn } from "@/common/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

interface Config {
  providers: {
    ollama_base_url: string;
    vllm_base_url: string;
    lmstudio_base_url: string;
    openai_api_key?: string;
    gemini_api_key?: string;
    openrouter_api_key?: string;
  };
  server: {
    port: number;
    log_level: string;
  };
}

export function Settings() {
  const [activeTab, setActiveTab] = useState<"preferences" | "providers" | "server">("preferences");
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [prefProvider, setPrefProvider] = useState("");
  const [prefModel, setPrefModel] = useState("");

  // Backend config state
  const [config, setConfig] = useState<Config | null>(null);
  const [saving, setSaving] = useState(false);
  const [appSaved, setAppSaved] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([listModels(), getConfig()])
      .then(([mList, cfg]) => {
        setModels(mList);
        setConfig(cfg);

        // Defaults
        const d = getDefaults();
        if (d) {
          setPrefProvider(d.provider);
          setPrefModel(d.model);
        } else if (mList.length > 0) {
          setPrefProvider(mList[0].provider);
          setPrefModel(mList[0].id ?? mList[0].name);
        }
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, []);

  const handleSavePreferences = () => {
    setDefaults(prefProvider, prefModel);
    setAppSaved(true);
    setTimeout(() => setAppSaved(false), 2000);
  };

  const handleSaveConfig = async () => {
    if (!config) return;
    setSaving(true);
    try {
      await updateConfig(config);
      setAppSaved(true);
      setTimeout(() => setAppSaved(false), 2000);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  if (loading)
    return (
      <div className="p-8">
        <Loader2 className="animate-spin h-8 w-8 text-emerald-500" />
      </div>
    );

  return (
    <div className="space-y-8 max-w-5xl">
      <div className="flex flex-col gap-2">
        <h2 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
          <SettingsIcon className="h-8 w-8 text-slate-400" />
          System Settings
        </h2>
        <p className="text-slate-400 mt-1">
          Configure model providers, cloud orchestration, and platform behavior.
        </p>
      </div>

      <div className="flex gap-1 p-1 bg-white/5 rounded-xl w-fit backdrop-blur-md">
        {(
          [
            { id: "preferences", label: "Preferences", icon: SettingsIcon },
            { id: "providers", label: "Provider Engine", icon: Cpu },
            { id: "server", label: "Network & Server", icon: Server },
          ] as const
        ).map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setActiveTab(tab.id)}
            className={cn(
              "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all",
              activeTab === tab.id
                ? "bg-emerald-600 text-white shadow-lg shadow-emerald-600/20"
                : "text-slate-400 hover:text-slate-200 hover:bg-white/5",
            )}
          >
            <tab.icon className="h-4 w-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/20 text-red-400 p-4 rounded-xl text-sm">
          {error}
        </div>
      )}

      {/* Tabs Content */}
      <div className="page-enter">
        {activeTab === "preferences" && (
          <Card className="border-white/5 bg-white/[0.02] backdrop-blur-md">
            <CardHeader>
              <CardTitle className="text-white">Default Configuration</CardTitle>
              <CardDescription>
                Target model and provider for immediate chat interaction.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <Label className="text-slate-300">Default Provider</Label>
                  <select
                    value={prefProvider}
                    onChange={(e) => setPrefProvider(e.target.value)}
                    className="w-full bg-slate-900 border-slate-700 text-sm text-white rounded-lg p-2.5 focus:ring-emerald-500 focus:border-emerald-500"
                  >
                    {Array.from(new Set(models.map((m) => m.provider))).map((p) => (
                      <option key={p} value={p}>
                        {p}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="space-y-2">
                  <Label className="text-slate-300">Default Model</Label>
                  <select
                    value={prefModel}
                    onChange={(e) => setPrefModel(e.target.value)}
                    className="w-full bg-slate-900 border-slate-700 text-sm text-white rounded-lg p-2.5 focus:ring-emerald-500 focus:border-emerald-500"
                  >
                    {models
                      .filter((m) => m.provider === prefProvider)
                      .map((m) => (
                        <option key={m.id ?? m.name} value={m.id ?? m.name}>
                          {m.name}
                        </option>
                      ))}
                  </select>
                </div>
              </div>
              <Button
                onClick={handleSavePreferences}
                className="bg-emerald-600 hover:bg-emerald-700"
              >
                {appSaved ? <Check className="h-4 w-4 mr-2" /> : <Save className="h-4 w-4 mr-2" />}
                {appSaved ? "Applied" : "Apply Defaults"}
              </Button>
            </CardContent>
          </Card>
        )}

        {activeTab === "providers" && (
          <div className="space-y-6">
            <Card className="border-white/5 bg-white/[0.02] backdrop-blur-md">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Cpu className="h-5 w-5 text-emerald-500" />
                  Local Provider Endpoints
                </CardTitle>
                <CardDescription>Ollama, vLLM, and LM Studio connectivity.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <Label className="text-slate-300">Ollama Base URL</Label>
                    <Input
                      value={config.providers.ollama_base_url}
                      onChange={(e) =>
                        setConfig({
                          ...config,
                          providers: { ...config.providers, ollama_base_url: e.target.value },
                        })
                      }
                      className="bg-slate-900 border-slate-700 text-white"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label className="text-slate-300">vLLM Base URL</Label>
                    <Input
                      value={config.providers.vllm_base_url}
                      onChange={(e) =>
                        setConfig({
                          ...config,
                          providers: { ...config.providers, vllm_base_url: e.target.value },
                        })
                      }
                      className="bg-slate-900 border-slate-700 text-white"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label className="text-slate-300">LM Studio URL</Label>
                    <Input
                      value={config.providers.lmstudio_base_url}
                      onChange={(e) =>
                        setConfig({
                          ...config,
                          providers: { ...config.providers, lmstudio_base_url: e.target.value },
                        })
                      }
                      className="bg-slate-900 border-slate-700 text-white"
                    />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-white/5 bg-white/[0.02] backdrop-blur-md">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Globe className="h-5 w-5 text-blue-500" />
                  Cloud & External Keys
                </CardTitle>
                <CardDescription>Securely bridge to remote LLM foundations.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <Label className="text-slate-300">OpenAI API Key</Label>
                    <Input
                      type="password"
                      placeholder="sk-..."
                      value={config.providers.openai_api_key || ""}
                      onChange={(e) =>
                        setConfig({
                          ...config,
                          providers: { ...config.providers, openai_api_key: e.target.value },
                        })
                      }
                      className="bg-slate-900 border-slate-700 text-white placeholder:text-slate-700"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label className="text-slate-300">Google Gemini Key</Label>
                    <Input
                      type="password"
                      value={config.providers.gemini_api_key || ""}
                      onChange={(e) =>
                        setConfig({
                          ...config,
                          providers: { ...config.providers, gemini_api_key: e.target.value },
                        })
                      }
                      className="bg-slate-900 border-slate-700 text-white"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label className="text-slate-300">OpenRouter API Key</Label>
                    <Input
                      type="password"
                      placeholder="sk-or-v1-..."
                      value={config.providers.openrouter_api_key || ""}
                      onChange={(e) =>
                        setConfig({
                          ...config,
                          providers: { ...config.providers, openrouter_api_key: e.target.value },
                        })
                      }
                      className="bg-slate-900 border-slate-700 text-white placeholder:text-slate-700"
                    />
                  </div>
                </div>
                <Button
                  onClick={handleSaveConfig}
                  disabled={saving}
                  className="bg-blue-600 hover:bg-blue-700"
                >
                  {saving ? (
                    <Loader2 className="animate-spin h-4 w-4 mr-2" />
                  ) : (
                    <Save className="h-4 w-4 mr-2" />
                  )}
                  {appSaved ? "Persisted" : "Persist to .env"}
                </Button>
              </CardContent>
            </Card>
          </div>
        )}

        {activeTab === "server" && (
          <Card className="border-white/5 bg-white/[0.02] backdrop-blur-md">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Server className="h-5 w-5 text-slate-400" />
                Network & Security
              </CardTitle>
              <CardDescription>Internal server bind configuration.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <Label className="text-slate-300">Server Port</Label>
                  <Input
                    type="number"
                    value={config.server.port}
                    onChange={(e) =>
                      setConfig({
                        ...config,
                        server: { ...config.server, port: parseInt(e.target.value, 10) },
                      })
                    }
                    className="bg-slate-900 border-slate-700 text-white"
                  />
                </div>
                <div className="space-y-2">
                  <Label className="text-slate-300">Log Level</Label>
                  <select
                    value={config.server.log_level}
                    onChange={(e) =>
                      setConfig({
                        ...config,
                        server: { ...config.server, log_level: e.target.value },
                      })
                    }
                    className="w-full bg-slate-900 border-slate-700 text-sm text-white rounded-lg p-2.5"
                  >
                    <option value="debug">Debug</option>
                    <option value="info">Info</option>
                    <option value="warning">Warning</option>
                    <option value="error">Error</option>
                  </select>
                </div>
              </div>
              <Button
                onClick={handleSaveConfig}
                disabled={saving}
                className="bg-slate-700 hover:bg-slate-600"
              >
                {saving ? (
                  <Loader2 className="animate-spin h-4 w-4 mr-2" />
                ) : (
                  <Save className="h-4 w-4 mr-2" />
                )}
                Save Server Config
              </Button>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
