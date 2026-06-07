import {
  BookOpen,
  Cpu,
  ExternalLink,
  GitBranch,
  HelpCircle,
  Layers,
  type LucideIcon,
  Server,
  Sparkles,
} from "lucide-react";
import { type ReactNode, useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import { cn } from "@/common/utils";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  fleetResources,
  type GuideTabId,
  guideTabs,
  hardwareTiers,
  modelLandscapeNotes,
  providerComparisonRows,
  providerGuides,
} from "@/content/local-llm-guide";

function ExternalAnchor({
  href,
  children,
  className,
}: {
  href: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className={cn(
        "inline-flex items-center gap-1.5 text-emerald-400 hover:text-emerald-300 underline-offset-4 hover:underline transition-colors",
        className,
      )}
    >
      {children}
      <ExternalLink className="w-3.5 h-3.5 shrink-0 opacity-70" />
    </a>
  );
}

function ProviderBadge({ badge }: { badge?: "recommended" | "advanced" | "catalog" }) {
  if (!badge) return null;
  const styles = {
    recommended: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
    advanced: "bg-amber-500/15 text-amber-300 border-amber-500/30",
    catalog: "bg-sky-500/15 text-sky-300 border-sky-500/30",
  };
  const labels = {
    recommended: "Fleet default path",
    advanced: "Power user",
    catalog: "Model catalog",
  };
  return (
    <span
      className={cn(
        "px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-widest rounded-full border",
        styles[badge],
      )}
    >
      {labels[badge]}
    </span>
  );
}

function SectionCard({
  title,
  description,
  children,
  className,
}: {
  title: string;
  description?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <Card
      className={cn(
        "bg-white/5 border-white/10 backdrop-blur-xl text-slate-100 shadow-none",
        className,
      )}
    >
      <CardHeader>
        <CardTitle className="text-xl text-white">{title}</CardTitle>
        {description ? (
          <CardDescription className="text-slate-400">{description}</CardDescription>
        ) : null}
      </CardHeader>
      <CardContent className="text-sm text-slate-300 leading-relaxed space-y-3">
        {children}
      </CardContent>
    </Card>
  );
}

function OverviewTab() {
  return (
    <div className="space-y-6">
      <SectionCard
        title="What is a local LLM?"
        description="Run language models on your own machine instead of sending prompts to a cloud API."
      >
        <p>
          A <strong className="text-white">local LLM stack</strong> has three layers: (1) model
          weights on disk, (2) a <strong className="text-white">runtime</strong> that loads them
          (Ollama, LM Studio, vLLM), and (3) <strong className="text-white">clients</strong> that
          send prompts (this dashboard, Cursor, fleet MCP servers).
        </p>
        <p>
          Benefits: privacy, no per-token cloud bill, works offline, predictable latency on a good
          GPU. Tradeoff: you own hardware sizing, updates, and model selection.
        </p>
      </SectionCard>

      <div className="grid md:grid-cols-2 gap-6">
        <SectionCard
          title="This hub (local-llm-mcp)"
          description="Optional control plane — port 10832/10833"
        >
          <ul className="list-disc pl-5 space-y-2">
            <li>One dashboard to see providers, chat, GPU stats, and fleet links</li>
            <li>
              MCP tools: <code className="text-emerald-300">llm_models</code>,{" "}
              <code className="text-emerald-300">llm_generation</code>, provider portmanteaus
            </li>
            <li>Live Settings for Ollama / LM Studio / vLLM URLs and API keys</li>
          </ul>
        </SectionCard>
        <SectionCard
          title="Direct Ollama (fleet default)"
          description="What most MCP repos use today"
        >
          <ul className="list-disc pl-5 space-y-2">
            <li>
              Peer servers set{" "}
              <code className="text-emerald-300">
                *_SAMPLING_BASE_URL=http://127.0.0.1:11434/v1
              </code>
            </li>
            <li>No dependency on this hub being running</li>
            <li>Simplest path for agentic workflows inside jellyfin-mcp, arxiv-mcp, etc.</li>
          </ul>
        </SectionCard>
      </div>

      <SectionCard
        title="Privacy"
        description="Zero telemetry from this dashboard to model vendors — when running locally"
      >
        <p>
          Prompts stay on your LAN when you use Ollama, LM Studio, or local vLLM. Cloud providers
          (OpenAI, Anthropic, Gemini) only see traffic you explicitly configure with API keys in
          Settings.
        </p>
      </SectionCard>
    </div>
  );
}

function HardwareTab() {
  return (
    <div className="space-y-6">
      <SectionCard
        title="VRAM is the bottleneck"
        description="System RAM matters for CPU offload; GPU VRAM decides which model fits."
      >
        <p>
          Pick a model that fits in VRAM at your chosen quantization. If it does not fit, the
          runtime spills to CPU RAM — usable for testing, painful for daily coding.
        </p>
      </SectionCard>

      <div className="grid gap-4">
        {hardwareTiers.map((tier) => (
          <div
            key={tier.name}
            className="p-5 rounded-2xl border border-white/10 bg-white/[0.03] hover:border-emerald-500/20 transition-colors"
          >
            <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
              <h3 className="text-lg font-bold text-white">{tier.name}</h3>
              <div className="flex gap-2">
                <Badge variant="outline" className="border-white/15 text-slate-300">
                  VRAM {tier.vram}
                </Badge>
                <Badge variant="outline" className="border-white/15 text-slate-300">
                  RAM {tier.ram}
                </Badge>
              </div>
            </div>
            <p className="text-emerald-300/90 text-sm font-medium mb-2">{tier.exampleModels}</p>
            <p className="text-slate-400 text-sm">{tier.notes}</p>
          </div>
        ))}
      </div>

      <SectionCard title="CPU-only and Apple Silicon">
        <p>
          Ollama and LM Studio run on CPU and integrated GPUs with reduced speed. NVIDIA CUDA still
          wins for large models. On laptops, favor 7B–14B quantizations before chasing 32B+ weights.
        </p>
      </SectionCard>
    </div>
  );
}

function ProvidersTab() {
  return (
    <div className="space-y-8">
      <SectionCard title="At a glance" description="How the four pillars relate">
        <div className="overflow-x-auto -mx-2">
          <table className="w-full min-w-[640px] text-left text-sm border-collapse">
            <thead>
              <tr className="border-b border-white/10 text-slate-400">
                <th className="py-3 pr-4 font-semibold">Dimension</th>
                <th className="py-3 px-2 font-semibold text-emerald-400">Ollama</th>
                <th className="py-3 px-2 font-semibold text-sky-400">LM Studio</th>
                <th className="py-3 px-2 font-semibold text-amber-400">vLLM</th>
                <th className="py-3 pl-2 font-semibold text-violet-400">Hugging Face</th>
              </tr>
            </thead>
            <tbody>
              {providerComparisonRows.map((row) => (
                <tr key={row.dimension} className="border-b border-white/5">
                  <td className="py-3 pr-4 text-slate-300">{row.dimension}</td>
                  <td className="py-3 px-2">{row.ollama}</td>
                  <td className="py-3 px-2">{row.lmstudio}</td>
                  <td className="py-3 px-2">{row.vllm}</td>
                  <td className="py-3 pl-2">{row.huggingface}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <div className="grid lg:grid-cols-2 gap-6">
        {providerGuides.map((p) => (
          <Card
            key={p.id}
            className="bg-gradient-to-br from-white/[0.06] to-transparent border-white/10 text-slate-100 shadow-none"
          >
            <CardHeader className="space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <CardTitle className="text-2xl text-white">{p.name}</CardTitle>
                <ProviderBadge badge={p.badge} />
              </div>
              <CardDescription className="text-slate-400">
                Port <span className="text-slate-200 font-mono">{p.defaultPort}</span>
                {" · "}
                {p.apiStyle}
              </CardDescription>
              <div className="flex flex-wrap gap-3 text-sm">
                <ExternalAnchor href={p.homepage}>Homepage</ExternalAnchor>
                {p.docsUrl ? <ExternalAnchor href={p.docsUrl}>Documentation</ExternalAnchor> : null}
              </div>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
              <div>
                <h4 className="text-xs font-bold uppercase tracking-widest text-emerald-400 mb-2">
                  Best for
                </h4>
                <ul className="list-disc pl-5 space-y-1 text-slate-300">
                  {p.bestFor.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
              <div>
                <h4 className="text-xs font-bold uppercase tracking-widest text-amber-400/90 mb-2">
                  Tradeoffs
                </h4>
                <ul className="list-disc pl-5 space-y-1 text-slate-400">
                  {p.tradeoffs.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
              <div>
                <h4 className="text-xs font-bold uppercase tracking-widest text-slate-400 mb-2">
                  Quick start
                </h4>
                <ol className="list-decimal pl-5 space-y-1 text-slate-300">
                  {p.setupSteps.map((step) => (
                    <li key={step}>{step}</li>
                  ))}
                </ol>
              </div>
              <p className="text-[11px] text-slate-500 pt-2 border-t border-white/5">
                Hub env: <code className="text-slate-400">{p.localLlmMcpEnv}</code>
              </p>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}

function ModelsTab() {
  return (
    <div className="space-y-6">
      <SectionCard
        title="Model landscape"
        description="Weights are files; runtimes decide how they are loaded and served."
      >
        <p>
          Start from your <strong className="text-white">hardware tier</strong>, then pick a
          runtime. Use{" "}
          <ExternalAnchor href="https://ollama.com/library">Ollama Library</ExternalAnchor> or{" "}
          <ExternalAnchor href="https://huggingface.co/models">Hugging Face Models</ExternalAnchor>{" "}
          to discover options. LM Studio wraps the same GGUF ecosystem with a GUI.
        </p>
      </SectionCard>

      <div className="grid md:grid-cols-2 gap-4">
        {modelLandscapeNotes.map((note) => (
          <div key={note.title} className="p-5 rounded-2xl border border-white/10 bg-white/[0.03]">
            <h3 className="font-bold text-white mb-2">{note.title}</h3>
            <p className="text-slate-400 text-sm leading-relaxed">{note.body}</p>
          </div>
        ))}
      </div>

      <SectionCard title="Suggested starting points (2026)">
        <ul className="space-y-3">
          <li>
            <span className="text-white font-medium">General + coding (16 GB VRAM):</span>{" "}
            <code className="text-emerald-300">qwen2.5-coder:7b</code>,{" "}
            <code className="text-emerald-300">llama3.2</code>,{" "}
            <code className="text-emerald-300">mistral</code> via Ollama
          </li>
          <li>
            <span className="text-white font-medium">Stronger local (24 GB+):</span> Qwen 32B class,
            DeepSeek distill, or MoE builds — verify fit in Performance tab
          </li>
          <li>
            <span className="text-white font-medium">Vision:</span> LLaVA / Qwen-VL variants when
            your provider exposes multimodal endpoints
          </li>
        </ul>
      </SectionCard>
    </div>
  );
}

function HubTab() {
  return (
    <div className="space-y-6">
      <SectionCard title="Ports and processes">
        <ul className="space-y-2 font-mono text-sm">
          <li>
            <span className="text-slate-500">Dashboard UI</span>{" "}
            <span className="text-white">http://127.0.0.1:10832</span>
          </li>
          <li>
            <span className="text-slate-500">Config API</span>{" "}
            <span className="text-white">http://127.0.0.1:10833</span>
          </li>
          <li>
            <span className="text-slate-500">MCP server</span>{" "}
            <span className="text-white">stdio via `uv run llm-mcp`</span>
          </li>
        </ul>
      </SectionCard>

      <SectionCard title="MCP portmanteau tools">
        <div className="grid sm:grid-cols-2 gap-3">
          {[
            ["llm_health", "Health, metrics, tool discovery"],
            ["llm_models", "List and register models across providers"],
            ["llm_generation", "Text, chat, embeddings"],
            ["llm_multimodal", "Vision when provider supports it"],
            ["llm_ollama", "Ollama-specific pull/list/chat"],
            ["llm_lmstudio", "LM Studio load and chat"],
            ["llm_vllm", "vLLM serving (when installed)"],
            ["llm_huggingface", "Hub search and transformers paths"],
          ].map(([tool, desc]) => (
            <div key={tool} className="p-3 rounded-xl bg-black/20 border border-white/5">
              <code className="text-emerald-300">{tool}</code>
              <p className="text-slate-500 text-xs mt-1">{desc}</p>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Configuration">
        <p>
          Use the <strong className="text-white">Settings</strong> page to set provider base URLs
          and cloud API keys. Values persist to your local{" "}
          <code className="text-slate-400">.env</code>. See{" "}
          <ExternalAnchor href="https://github.com/sandraschi/local-llm-mcp/blob/master/.env.example">
            .env.example
          </ExternalAnchor>{" "}
          in the repo for all keys.
        </p>
      </SectionCard>
    </div>
  );
}

function FleetTab() {
  return (
    <div className="space-y-6">
      <SectionCard
        title="Fleet inference pattern"
        description="Most Sandra MCP servers bypass this hub and talk to Ollama directly."
      >
        <p className="mb-4">Server-side agentic tools use an OpenAI-compatible URL, typically:</p>
        <pre className="p-4 rounded-xl bg-black/30 border border-white/10 text-emerald-300 text-xs overflow-x-auto">
          {`JELLYFIN_SAMPLING_BASE_URL=http://127.0.0.1:11434/v1
ARXIV_MCP_SAMPLING_BASE_URL=http://127.0.0.1:11434/v1`}
        </pre>
        <p className="mt-4">
          Keep Ollama running on <code className="text-slate-300">11434</code> for fleet-wide
          sampling. Use this hub when you want centralized model management, multi-provider routing,
          or the dashboard fleet launcher.
        </p>
      </SectionCard>

      <SectionCard title="Further reading">
        <ul className="space-y-4">
          {fleetResources.map((r) => (
            <li key={r.url}>
              <ExternalAnchor href={r.url}>{r.name}</ExternalAnchor>
              <p className="text-slate-500 text-sm mt-1">{r.blurb}</p>
            </li>
          ))}
        </ul>
      </SectionCard>
    </div>
  );
}

const tabIcons: Record<GuideTabId, LucideIcon> = {
  overview: BookOpen,
  hardware: Cpu,
  providers: Layers,
  models: Sparkles,
  hub: Server,
  fleet: GitBranch,
};

export function Help() {
  const [searchParams, setSearchParams] = useSearchParams();
  const tabParam = searchParams.get("tab") as GuideTabId | null;
  const activeTab = useMemo(() => {
    if (tabParam && guideTabs.some((t) => t.id === tabParam)) return tabParam;
    return "overview" satisfies GuideTabId;
  }, [tabParam]);

  const onTabChange = (value: string) => {
    setSearchParams({ tab: value }, { replace: true });
  };

  return (
    <div className="p-6 md:p-8 max-w-7xl mx-auto animate-in fade-in duration-500">
      <header className="mb-8 flex flex-col md:flex-row md:items-center gap-4 md:gap-6">
        <div className="p-3 bg-emerald-500/10 rounded-2xl border border-emerald-500/20 w-fit">
          <HelpCircle className="w-8 h-8 text-emerald-400" />
        </div>
        <div>
          <h1 className="text-3xl md:text-4xl font-bold text-white tracking-tight">
            Local LLM Guide
          </h1>
          <p className="text-slate-400 mt-1 max-w-2xl">
            Hardware, providers, models, and how this hub fits the fleet — with links to official
            docs.
          </p>
        </div>
      </header>

      <Tabs value={activeTab} onValueChange={onTabChange} className="w-full">
        <TabsList className="flex flex-wrap h-auto gap-1 p-1.5 bg-white/5 border border-white/10 rounded-2xl mb-6 w-full justify-start">
          {guideTabs.map((tab) => {
            const Icon = tabIcons[tab.id];
            return (
              <TabsTrigger
                key={tab.id}
                value={tab.id}
                className="data-[state=active]:bg-emerald-600/20 data-[state=active]:text-emerald-100 data-[state=active]:border-emerald-500/30 border border-transparent rounded-xl px-4 py-2.5 text-slate-400 gap-2"
              >
                <Icon className="w-4 h-4 shrink-0" />
                <span className="hidden sm:inline">{tab.label}</span>
                <span className="sm:hidden">{tab.label.split(" ")[0]}</span>
              </TabsTrigger>
            );
          })}
        </TabsList>

        {guideTabs.map((tab) => (
          <TabsContent key={tab.id} value={tab.id} className="mt-0 focus-visible:ring-0">
            <p className="text-sm text-slate-500 mb-6 pl-1">{tab.description}</p>
            {tab.id === "overview" && <OverviewTab />}
            {tab.id === "hardware" && <HardwareTab />}
            {tab.id === "providers" && <ProvidersTab />}
            {tab.id === "models" && <ModelsTab />}
            {tab.id === "hub" && <HubTab />}
            {tab.id === "fleet" && <FleetTab />}
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}
