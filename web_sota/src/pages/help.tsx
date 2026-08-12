import {
  AlertTriangle,
  ArrowRight,
  BookOpen,
  Bot,
  Boxes,
  Cpu,
  Database,
  HelpCircle,
  Layers,
  LifeBuoy,
  Network,
  Server,
  Sparkles,
  Terminal,
  Wrench,
} from "lucide-react";
import type React from "react";
import { useState } from "react";

type TabId =
  | "overview"
  | "engines"
  | "glimmer"
  | "supervision"
  | "providers"
  | "models"
  | "troubleshooting"
  | "fleet"
  | "faq";

const TABS: { id: TabId; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { id: "overview", label: "Overview", icon: BookOpen },
  { id: "engines", label: "Engines", icon: Server },
  { id: "glimmer", label: "Muse Glimmer", icon: Sparkles },
  { id: "supervision", label: "Engine Supervision", icon: Wrench },
  { id: "providers", label: "Providers", icon: Network },
  { id: "models", label: "Models", icon: Boxes },
  { id: "troubleshooting", label: "Troubleshooting", icon: AlertTriangle },
  { id: "fleet", label: "Fleet", icon: Layers },
  { id: "faq", label: "FAQ", icon: HelpCircle },
];

function Code({ children }: { children: React.ReactNode }) {
  return (
    <code className="px-1.5 py-0.5 bg-zinc-800 text-emerald-300 rounded text-[0.85em] font-mono">
      {children}
    </code>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-8">
      <h2 className="text-xl font-bold text-white mb-3 flex items-center gap-2">
        <span className="w-1.5 h-5 bg-blue-500 rounded-full" />
        {title}
      </h2>
      <div className="text-gray-300 leading-relaxed space-y-3">{children}</div>
    </div>
  );
}

function InfoBox({
  title,
  children,
  tone = "blue",
}: {
  title: string;
  children: React.ReactNode;
  tone?: "blue" | "amber" | "red" | "green";
}) {
  const tones = {
    blue: "bg-blue-500/10 border-blue-500/30 text-blue-200",
    amber: "bg-amber-500/10 border-amber-500/30 text-amber-200",
    red: "bg-red-500/10 border-red-500/30 text-red-200",
    green: "bg-emerald-500/10 border-emerald-500/30 text-emerald-200",
  };
  return (
    <div className={`p-4 rounded-2xl border ${tones[tone]} mb-4`}>
      <div className="font-bold mb-1">{title}</div>
      <div className="text-sm leading-relaxed opacity-90">{children}</div>
    </div>
  );
}

function PortTable() {
  const rows = [
    [
      "11435",
      "Truncating proxy",
      "The endpoint everything talks to (opencode, fleet servers). Trims oversized requests.",
    ],
    [
      "11439",
      "llama-server",
      "Muse Glimmer 30B inference (131K context, full GPU, DFlash drafter).",
    ],
    ["11434", "Ollama", "Ollama daemon (started on demand; not running while Glimmer holds VRAM)."],
  ];
  return (
    <div className="overflow-x-auto rounded-2xl border border-white/10">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-white/5 text-left text-gray-400">
            <th className="px-4 py-2.5 font-semibold">Port</th>
            <th className="px-4 py-2.5 font-semibold">Service</th>
            <th className="px-4 py-2.5 font-semibold">Role</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r[0]} className="border-t border-white/5">
              <td className="px-4 py-2.5 font-mono text-blue-300">{r[0]}</td>
              <td className="px-4 py-2.5 font-semibold text-gray-200">{r[1]}</td>
              <td className="px-4 py-2.5 text-gray-400">{r[2]}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function OverviewTab() {
  return (
    <div>
      <Section title="What is this server?">
        <p>
          <strong className="text-white">Local LLM MCP</strong> is the fleet's control plane for
          local inference. It supervises the engines that run on your GPU (Muse Glimmer via
          llama.cpp, Ollama), manages their models, routes OpenAI-compatible traffic, and reports
          health -- so your coding agents, news pipelines, and research jobs all share one inference
          stack instead of fighting over the 4090.
        </p>
      </Section>
      <Section title="Quick start">
        <ol className="list-decimal pl-5 space-y-2">
          <li>
            Start the engine: <Code>llm_engine(operation="start", engine="llama")</Code>
          </li>
          <li>
            Check everything: <Code>llm_engine(operation="status")</Code> -- processes, ports, VRAM,
            loaded models
          </li>
          <li>
            Point any OpenAI-compatible client at <Code>http://127.0.0.1:11435/v1</Code>
          </li>
          <li>
            In opencode, pick <Code>local-llama / muse-glimmer-30b</Code>
          </li>
        </ol>
      </Section>
      <Section title="Architecture">
        <PortTable />
        <p className="text-sm text-gray-400">
          The proxy on 11435 is the front door. It exists because opencode sends enormous requests
          (full session history + a 409-tool catalog = ~200K tokens) that no 24 GB GPU can hold at
          once. The proxy trims, filters, and pins the important parts before they reach the server.
          See the Muse Glimmer tab for the full story.
        </p>
      </Section>
    </div>
  );
}

function EnginesTab() {
  return (
    <div>
      <Section title="Two engines, one GPU">
        <p>
          This machine runs <strong className="text-white">two inference engines</strong> that share
          the RTX 4090's 24 GB of VRAM -- and they do <em>not</em> share politely.
        </p>
      </Section>
      <InfoBox title="The one-way door (VRAM contention)" tone="red">
        Muse Glimmer holds ~21 GB of VRAM while running. Ollama cannot evict it (they are separate
        processes), so an Ollama model loaded alongside Glimmer runs mostly on CPU at 1-3
        tokens/sec. They are mutually exclusive: pick one engine per session. Stop Glimmer before
        using Ollama, and vice versa -- <Code>llm_engine(operation="stop", engine="...")</Code>.
      </InfoBox>
      <div className="grid md:grid-cols-2 gap-4 mb-4">
        <div className="p-5 rounded-2xl bg-white/5 border border-white/10">
          <div className="flex items-center gap-2 mb-2">
            <Bot className="w-5 h-5 text-blue-400" />
            <h3 className="font-bold text-white">Muse Glimmer 30B (llama.cpp)</h3>
          </div>
          <p className="text-sm text-gray-400">
            The flagship. Agentic, multimodal, 131K context, speculative decoding (~40 tok/s).
            Served natively by llama.cpp because no release build supports its quantization yet.
          </p>
          <p className="text-xs text-gray-500 mt-2">
            Ports <Code>11439</Code> (server) + <Code>11435</Code> (proxy)
          </p>
        </div>
        <div className="p-5 rounded-2xl bg-white/5 border border-white/10">
          <div className="flex items-center gap-2 mb-2">
            <Database className="w-5 h-5 text-emerald-400" />
            <h3 className="font-bold text-white">Ollama</h3>
          </div>
          <p className="text-sm text-gray-400">
            The classic local stack (Qwen, Gemma, DeepSeek). Manages its own model lifecycle and
            serves OpenAI-compatible /v1 plus its native /api.
          </p>
          <p className="text-xs text-gray-500 mt-2">
            Port <Code>11434</Code>
          </p>
        </div>
      </div>
      <Section title="Why is Glimmer not in Ollama?">
        <p>
          Ollama bundles a specific llama.cpp version. Muse Glimmer uses a <em>kquant</em> GGUF
          format newer than anything Ollama (or any llama.cpp release) could load on launch day. The
          fleet compiled llama.cpp from source with CUDA and runs it natively -- this is also the
          faster path, since it skips Ollama's overhead entirely.
        </p>
      </Section>
    </div>
  );
}

function GlimmerTab() {
  return (
    <div>
      <Section title="Meet Muse Glimmer 30B">
        <p>
          <strong className="text-white">Muse Glimmer</strong> is Meta's agentic assistant model
          (Apache 2.0, August 2026). It is a <em>multimodal reasoning agent</em>: it reads
          screenshots, charts, and documents, plans multi-step tasks, calls tools, recovers from
          failures -- all locally on your GPU.
        </p>
        <ul className="list-disc pl-5 space-y-1.5">
          <li>
            <strong className="text-white">30B parameters</strong> -- a serious model, quantized to
            fit 24 GB with room for context.
          </li>
          <li>
            <strong className="text-white">131,072-token context</strong> -- long sessions, big
            codebases, whole documents.
          </li>
          <li>
            <strong className="text-white">Multimodal</strong> -- text + images in, text out.
          </li>
          <li>
            <strong className="text-white">DFlash drafter</strong> -- a speculative-decoding
            companion that proposes tokens in blocks of 16, roughly tripling generation speed (~40
            tok/s on the 4090).
          </li>
          <li>
            <strong className="text-white">Reasoning strength</strong> -- low/medium/high/xhigh;
            thoughts are extracted separately from answers.
          </li>
        </ul>
      </Section>
      <Section title="Using it from opencode">
        <p>
          The model is registered as the <Code>local-llama</Code> provider with model
          <Code>muse-glimmer-30b</Code> (shown as "Muse Glimmer 30B (Goliath)"). Select it in the
          model picker. Reasoning arrives as thinking, then the answer; tool calls go through the
          core whitelist (bash, read, write, edit, glob, grep, ...).
        </p>
      </Section>
      <Section title="Why is there a proxy in the middle?">
        <p>
          opencode ignores context limits for custom providers. A heavy session sends ~200K tokens
          (history + its 409-tool MCP catalog) -- far beyond the 131K window. The proxy performs
          four surgeries on every request:
        </p>
        <ol className="list-decimal pl-5 space-y-1.5">
          <li>
            <strong className="text-white">Truncates the system prompt</strong> -- the fleet
            instruction wall triggers Glimmer's "agent mode" (it replies "Ready." instead of
            answering).
          </li>
          <li>
            <strong className="text-white">Pins your last message</strong> -- so history trimming
            can never eat your actual question.
          </li>
          <li>
            <strong className="text-white">Counts and trims tools</strong> -- 409 schemas (~102K
            tokens) are filtered to a core whitelist so the model still has real powers.
          </li>
          <li>
            <strong className="text-white">Tells the model honestly</strong> what it can and cannot
            do, so it never announces actions it cannot take.
          </li>
        </ol>
      </Section>
      <Section title="What you might see and what it means">
        <div className="overflow-x-auto rounded-2xl border border-white/10">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-white/5 text-left text-gray-400">
                <th className="px-4 py-2.5 font-semibold">Symptom</th>
                <th className="px-4 py-2.5 font-semibold">Meaning</th>
              </tr>
            </thead>
            <tbody>
              {[
                [
                  '"Ready." / "Context loaded." / "What task should I run?"',
                  "Glimmer thinks it is an agent harness, not a chat. Usually the proxy system-truncation is disabled or a new prompt shape slipped past it.",
                ],
                [
                  "Empty answer after thinking",
                  "Reasoning consumed the whole token budget, or the question was trimmed away. Send a shorter, direct prompt.",
                ],
                [
                  "Tools don't work",
                  "The 409-tool catalog exceeded the budget and tools were dropped. The proxy keeps core tools (bash etc.) -- MCP-server tools are the first to go.",
                ],
                [
                  "Very slow / 1-3 tok/s",
                  "Glimmer is not loaded; something else (Ollama model) holds the GPU, or the server is CPU-offloading.",
                ],
              ].map((row) => (
                <tr key={row[0]} className="border-t border-white/5">
                  <td className="px-4 py-2.5 text-gray-200">{row[0]}</td>
                  <td className="px-4 py-2.5 text-gray-400">{row[1]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
      <Section title="Live status and controls">
        <p>
          The <strong className="text-white">Glimmer</strong> page shows live engine state
          (llama-server, proxy, Ollama, GPU VRAM, loaded models) and can restart the engine:
        </p>
        <p className="mt-2">
          <a
            href="/glimmer"
            className="inline-flex items-center gap-2 rounded-xl bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500"
          >
            Open Glimmer status page <ArrowRight className="h-4 w-4" />
          </a>
        </p>
        <p className="mt-2 text-sm text-gray-400">
          Restarting llama-server reloads roughly 21 GB into VRAM and takes minutes; Ollama is
          stopped while Glimmer holds the card.
        </p>
      </Section>
    </div>
  );
}

function SupervisionTab() {
  return (
    <div>
      <Section title="The llm_engine tool">
        <p>
          One portmanteau tool supervises both engines and their models. It replaced the previous
          state of affairs where nothing could start, stop, or inspect the llama server at all.
        </p>
      </Section>
      <div className="overflow-x-auto rounded-2xl border border-white/10 mb-4">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-white/5 text-left text-gray-400">
              <th className="px-4 py-2.5 font-semibold">Operation</th>
              <th className="px-4 py-2.5 font-semibold">What it does</th>
              <th className="px-4 py-2.5 font-semibold">Example</th>
            </tr>
          </thead>
          <tbody>
            {[
              [
                "status",
                "Live state of both engines: processes, ports, health, loaded models, per-engine VRAM",
                'llm_engine(operation="status")',
              ],
              [
                "start",
                "Launch an engine (ollama or llama)",
                'llm_engine(operation="start", engine="llama")',
              ],
              [
                "stop",
                "Stop an engine and its child processes",
                'llm_engine(operation="stop", engine="ollama")',
              ],
              [
                "list_models",
                "List available models (Ollama) or loaded (llama)",
                'llm_engine(operation="list_models", engine="all")',
              ],
              [
                "load_model",
                "Load a model into memory (Ollama) or start the server (llama)",
                'llm_engine(operation="load_model", engine="ollama", model="qwen3.6:27b")',
              ],
              [
                "unload_model",
                "Free a model from memory",
                'llm_engine(operation="unload_model", engine="ollama", model="qwen3.6:27b")',
              ],
            ].map((row) => (
              <tr key={row[0]} className="border-t border-white/5">
                <td className="px-4 py-2.5 font-mono text-emerald-300">{row[0]}</td>
                <td className="px-4 py-2.5 text-gray-300">{row[1]}</td>
                <td className="px-4 py-2.5 font-mono text-xs text-gray-500">{row[2]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <InfoBox title="Why the process scan uses PowerShell" tone="blue">
        psutil's process enumeration hangs on this machine (a system process blocks it). The tool
        scans via PowerShell CIM instead -- same result, no hang. This is also why it is
        Windows-only for process operations.
      </InfoBox>
    </div>
  );
}

function ProvidersTab() {
  return (
    <div>
      <Section title="Provider model">
        <p>
          <Code>config.yaml</Code> at the repo root declares providers. The key one is
          <Code>local-llama</Code>: an OpenAI-compatible endpoint pointing at the Glimmer proxy (
          <Code>http://127.0.0.1:11435/v1</Code>) with the full model spec (131K context, image
          input, 4K max output).
        </p>
      </Section>
      <Section title="OpenAI-compatible gateway">
        <p>
          The server exposes <Code>/v1/chat/completions</Code> and routes by provider. Local engines
          (Ollama, llama.cpp) are detected on their ports; cloud providers (Anthropic, DeepSeek,
          ...) route to their APIs. Health checks per provider appear in{" "}
          <Code>GET /api/v1/health</Code>.
        </p>
      </Section>
      <Section title="Environment">
        <p>
          Key overrides live in <Code>.env</Code> (see <Code>.env.example</Code>):{" "}
          <Code>LLM_MCP_OLLAMA_PORT</Code>, <Code>LLM_MCP_LLAMA_SERVER_PORT</Code> (11439),{" "}
          <Code>LLM_MCP_LLAMA_PROXY_PORT</Code> (11435), and the llama server paths.
        </p>
      </Section>
    </div>
  );
}

function ModelsTab() {
  return (
    <div>
      <Section title="Ollama models (llm_ollama)">
        <p>
          Manage the Ollama catalog: <Code>list_models</Code>, <Code>load_model</Code>,{" "}
          <Code>unload_model</Code>, <Code>pull_model</Code>, <Code>delete_model</Code>.
        </p>
      </Section>
      <InfoBox title="num_ctx -- the silent tool-killer" tone="amber">
        Ollama defaults to a 2048-token context window. Agents send huge prompts (tools + history),
        Ollama silently truncates the tail -- which is where the tool definitions live -- and the
        model "never uses tools". The fleet fixes this per-model with
        <Code>options.num_ctx = 32768</Code> in opencode's provider config. If tools stop working on
        an Ollama model, check this first.
      </InfoBox>
      <Section title="Glimmer models (llama server)">
        <p>
          The llama server holds exactly one model per launch (currently Muse Glimmer 30B).
          <Code>load_model</Code> with the llama engine starts the server; <Code>unload_model</Code>{" "}
          stops it. Listing models returns the loaded GGUF.
        </p>
      </Section>
    </div>
  );
}

function TroubleshootingTab() {
  return (
    <div>
      <Section title="The classic failure modes">
        {[
          [
            "Gobbledigook / token soup output",
            "The chat template failed to load (the GGUF embeds a custom Onyx template). Fixed by extracting it and passing --chat-template-file. If you see raw <|message|> tokens, this is back.",
          ],
          [
            "Endless thinking, never answers",
            "The template forces high reasoning strength. The server caps reasoning with --reasoning-budget 1024.",
          ],
          [
            "Request too large / nothing happens",
            "Sessions grow past 131K tokens. The proxy trims history and tools; if it still fails, the session is beyond recovery -- start a fresh one.",
          ],
          [
            '"No response" after thinking',
            "The last user message was being trimmed away. Fixed by the proxy's last-message pin. If it recurs, your prompt is genuinely enormous.",
          ],
          [
            "GPU starved / slow everywhere",
            'Two engines fighting over VRAM. Stop one: llm_engine(operation="stop", ...). Long runners (aiwatcher, arxiv epistemic jobs) now share Glimmer via :11435 instead of starving.',
          ],
          [
            "Ollama model ignores tools",
            "num_ctx truncation (see Models tab). Set options.num_ctx to match the model's window.",
          ],
        ].map(([title, body]) => (
          <div key={title} className="p-4 rounded-2xl bg-white/5 border border-white/10 mb-3">
            <div className="font-bold text-gray-200 mb-1">{title}</div>
            <div className="text-sm text-gray-400">{body}</div>
          </div>
        ))}
      </Section>
    </div>
  );
}

function FleetTab() {
  return (
    <div>
      <Section title="Who shares the GPU">
        <p>
          Glimmer is not a museum piece -- the fleet's long-running jobs call it like any
          OpenAI-compatible endpoint. That converts VRAM contention into shared inference:
        </p>
        <ul className="list-disc pl-5 space-y-1.5">
          <li>
            <strong className="text-white">opencode</strong> --{" "}
            <Code>local-llama/muse-glimmer-30b</Code> (via proxy :11435)
          </li>
          <li>
            <strong className="text-white">aiwatcher</strong> (distillation, digest, Fritz) --{" "}
            <Code>LLM_BASE_URL=http://127.0.0.1:11435/v1</Code>
          </li>
          <li>
            <strong className="text-white">arxiv-mcp</strong> (deep epistemic analysis) --{" "}
            <Code>ARXIV_MCP_SAMPLING_BASE_URL=...:11435/v1</Code>, model{" "}
            <Code>muse-glimmer-30b</Code>
          </li>
          <li>
            <strong className="text-white">local-llm-mcp itself</strong> -- provider config +{" "}
            <Code>llm_engine</Code> supervision
          </li>
        </ul>
      </Section>
      <Section title="Serving playbook">
        <p>
          The full pattern for running a bleeding-edge model natively is documented in{" "}
          <Code>mcp-central-docs/patterns/LLAMA_CPP_NATIVE_MODEL_SERVING.md</Code> -- including the
          CUDA build, the cudart DLL gotcha, and the four proxy guardrails.
        </p>
      </Section>
    </div>
  );
}

function FaqTab() {
  const faqs: [string, React.ReactNode][] = [
    [
      'Why does the model say "Ready." instead of answering?',
      <>
        Glimmer is trained as an agent. A big system prompt (the fleet instructions) flips it into
        agent mode. The proxy truncates the system prompt and pins your question so it stays in chat
        mode. It is also told -- honestly -- which tools it actually has.
      </>,
    ],
    [
      "Why is there a random Python process between me and the model?",
      <>
        That is the truncating proxy. It is what makes opencode work with a 131K model at all.
        Without it, your session's 200K tokens would just be rejected.
      </>,
    ],
    [
      "Can I run Glimmer and Ollama at the same time?",
      <>
        No. 24 GB of VRAM fits one. The <Code>llm_engine</Code> status tool shows you the VRAM
        picture so you never guess.
      </>,
    ],
    [
      "Why is Glimmer not in Ollama?",
      <>
        Its quantization format (kquant) is newer than what Ollama bundles. The fleet runs it
        natively via llama.cpp compiled from source -- faster anyway.
      </>,
    ],
    [
      "What does the 11435 vs 11439 split mean?",
      <>
        <Code>11439</Code> is the raw llama-server. <Code>11435</Code> is the proxy -- always talk
        to 11435.
      </>,
    ],
    ["Does it cost anything?", <>No. Apache 2.0 model, local GPU, zero telemetry.</>],
  ];
  return (
    <div>
      <Section title="Questions from people who just met Glimmer">
        {faqs.map(([q, a]) => (
          <div key={q} className="p-4 rounded-2xl bg-white/5 border border-white/10 mb-3">
            <div className="font-bold text-white mb-1 flex items-start gap-2">
              <HelpCircle className="w-4 h-4 text-blue-400 mt-0.5 shrink-0" />
              {q}
            </div>
            <div className="text-sm text-gray-400 pl-6">{a}</div>
          </div>
        ))}
      </Section>
    </div>
  );
}

export const Help: React.FC = () => {
  const [active, setActive] = useState<TabId>("overview");

  const render = () => {
    switch (active) {
      case "overview":
        return <OverviewTab />;
      case "engines":
        return <EnginesTab />;
      case "glimmer":
        return <GlimmerTab />;
      case "supervision":
        return <SupervisionTab />;
      case "providers":
        return <ProvidersTab />;
      case "models":
        return <ModelsTab />;
      case "troubleshooting":
        return <TroubleshootingTab />;
      case "fleet":
        return <FleetTab />;
      case "faq":
        return <FaqTab />;
    }
  };

  return (
    <div data-testid="help-page" className="p-8 max-w-7xl mx-auto animate-in fade-in duration-700">
      <div className="mb-8 flex items-center gap-4">
        <div className="p-3 bg-blue-500/10 rounded-2xl border border-blue-500/20">
          <LifeBuoy className="w-8 h-8 text-blue-400" />
        </div>
        <div>
          <h1 className="text-4xl font-bold text-white tracking-tight">Help & Documentation</h1>
          <p className="text-gray-400 mt-1">
            Your local inference stack, explained -- engines, models, and the proxy that makes it
            all work.
          </p>
        </div>
      </div>

      <div
        data-testid="help-tabs"
        className="flex gap-2 overflow-x-auto pb-2 mb-8 border-b border-white/10"
      >
        {TABS.map((t) => {
          const Icon = t.icon;
          const activeTab = active === t.id;
          return (
            <button
              key={t.id}
              type="button"
              data-testid={`help-tab-${t.id}`}
              onClick={() => setActive(t.id)}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-t-xl text-sm font-semibold whitespace-nowrap transition-colors border-b-2 ${
                activeTab
                  ? "text-blue-300 border-blue-500 bg-blue-500/10"
                  : "text-gray-400 border-transparent hover:text-gray-200 hover:bg-white/5"
              }`}
            >
              <Icon className="w-4 h-4" />
              {t.label}
            </button>
          );
        })}
      </div>

      <div data-testid="help-content" className="max-w-4xl">
        {render()}
      </div>

      <div className="mt-12 p-6 rounded-3xl bg-gradient-to-r from-blue-600/20 to-indigo-600/20 border border-blue-500/30 flex flex-col md:flex-row items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white mb-1">Still confused?</h2>
          <p className="text-gray-300 text-sm">
            The fleet's serving playbook and model card live in mcp-central-docs. The
            <Code>llm_engine</Code> tool answers "is it running?" before you ask.
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs text-gray-400">
          <Cpu className="w-4 h-4" />
          <span>RTX 4090 - 24 GB</span>
          <Terminal className="w-4 h-4 ml-3" />
          <span>:11435</span>
        </div>
      </div>
    </div>
  );
};
