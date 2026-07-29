import { Bot, Check, Copy, Download, Loader2, Send, Sparkles, Trash2, User } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { getChatPrefs, setChatPrefs } from "@/api/chat-prefs";
import { generate, listModels, type ModelInfo } from "@/api/client";
import { getDefaults } from "@/api/defaults";
import { getPersonality, PERSONALITIES } from "@/common/personalities";
import { initSpeechService } from "@/common/speech-service";
import { cn } from "@/common/utils";
import { MicButton } from "@/components/MicButton";
import { ModelSelector } from "@/components/models/ModelSelector";
import { SpeakButton } from "@/components/SpeakButton";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

const REFINE_PROMPT_PREFIX =
  "Rewrite the following into a single clear, self-contained prompt for an LLM. Preserve the user's intent. Output only the rewritten prompt, no commentary.\n\n";

const HISTORY_KEY = "local-llm-chat-history";
const MAX_HISTORY = 100;

const EXAMPLE_PROMPTS = [
  {
    group: "Models",
    prompts: [
      "List all available models",
      "Compare the latest Llama and Qwen models",
      "Show me trending models on Hugging Face",
    ],
  },
  {
    group: "Chat",
    prompts: [
      "Help me write a Python script for data analysis",
      "Explain how transformer attention works",
      "Summarize the key concepts of RAG",
    ],
  },
  {
    group: "System",
    prompts: [
      "Check the health of all LLM providers",
      "Show me connected provider status",
      "Refresh model list and detect new models",
    ],
  },
];

type Message = { id: string; role: "user" | "assistant"; content: string };

const PERSONALITY_COLORS: Record<string, string> = {
  slate: "border-slate-500 bg-slate-500/10",
  amber: "border-amber-500 bg-amber-500/10",
  violet: "border-violet-500 bg-violet-500/10",
  emerald: "border-emerald-500 bg-emerald-500/10",
  blue: "border-blue-500 bg-blue-500/10",
  rose: "border-rose-500 bg-rose-500/10",
  cyan: "border-cyan-500 bg-cyan-500/10",
  orange: "border-orange-500 bg-orange-500/10",
  fuchsia: "border-fuchsia-500 bg-fuchsia-500/10",
  lime: "border-lime-500 bg-lime-500/10",
  indigo: "border-indigo-500 bg-indigo-500/10",
  teal: "border-teal-500 bg-teal-500/10",
};

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      onClick={() => {
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      }}
      className="p-1.5 rounded text-slate-400 hover:text-white hover:bg-white/10 transition-colors"
      title="Copy"
    >
      {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
    </button>
  );
}

function loadHistory(): Message[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function Chat() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedProvider, setSelectedProvider] = useState<string | undefined>();
  const [messages, setMessages] = useState<Message[]>(() => loadHistory());
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [refining, setRefining] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [skillName, setSkillName] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const prefs = getChatPrefs();
  const [personalityId, setPersonalityId] = useState(prefs.personalityId);
  const [promptRefinement, setPromptRefinement] = useState(prefs.promptRefinement);

  const personality =
    getPersonality(personalityId) ?? getPersonality("neutral") ?? PERSONALITIES[0];
  const personalityColor = PERSONALITY_COLORS[personality.color] ?? PERSONALITY_COLORS.slate;

  // Persist messages to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(messages.slice(-MAX_HISTORY)));
    } catch {
      /* ignore */
    }
  }, [messages]);

  // Skill fetch
  useEffect(() => {
    (async () => {
      try {
        const r = await fetch("/api/skills");
        if (r.ok) {
          const data = await r.json();
          const skills = data?.skills ?? [];
          if (skills.length > 0) setSkillName(skills[0].name || skills[0]);
        }
      } catch {
        /* no skills */
      }
    })();
  }, []);

  useEffect(() => {
    listModels()
      .then((list) => {
        setModels(list);
        const d = getDefaults();
        if (d && list.some((m) => (m.id ?? m.name) === d.model && m.provider === d.provider)) {
          setSelectedModel(d.model);
          setSelectedProvider(d.provider);
        } else if (list.length > 0 && !selectedModel) {
          const first = list[0];
          setSelectedModel(first.id ?? first.name);
          setSelectedProvider(first.provider);
        }
      })
      .catch(() => setModels([]));
  }, [selectedModel]);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Init speech service (probes speech-mcp, falls back to Web Speech API)
  useEffect(() => {
    initSpeechService();
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;
    if (!selectedModel) {
      setError("Select a model first.");
      return;
    }
    setError(null);
    setInput("");
    const userMsg: Message = { id: crypto.randomUUID(), role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    let promptToSend = text;
    if (promptRefinement) {
      setRefining(true);
      try {
        const refined = await generate(
          REFINE_PROMPT_PREFIX + text,
          selectedModel,
          selectedProvider,
        );
        promptToSend = refined.text.trim();
      } catch {
        // use original if refinement fails
      }
      setRefining(false);
    }

    const fullPrompt = personality.systemPrompt + promptToSend;

    try {
      const res = await generate(fullPrompt, selectedModel, selectedProvider);
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: "assistant", content: res.text },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: `Error: ${err instanceof Error ? err.message : String(err)}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function clearChat() {
    setMessages([]);
    setError(null);
    localStorage.removeItem(HISTORY_KEY);
  }

  function exportChat() {
    if (messages.length === 0) return;
    const lines = messages.map(
      (m) => `[${new Date().toISOString()}] ${m.role === "user" ? "You" : "AI"}: ${m.content}`,
    );
    const blob = new Blob([lines.join("\n\n---\n\n")], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `local-llm-chat-${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div data-testid="chat-page" className="flex h-[calc(100vh-8rem)] flex-col space-y-4">
      <div
        data-testid="chat-controls"
        className="flex flex-wrap items-center justify-between gap-4"
      >
        <div className="flex items-center gap-3">
          <h2 className="text-2xl font-bold tracking-tight text-white">Chat</h2>
          {skillName && (
            <span className="text-[10px] text-zinc-500 bg-zinc-800 px-1.5 py-0.5 rounded font-mono">
              skill:{skillName}
            </span>
          )}
        </div>
        <div className="flex items-center gap-3 flex-wrap">
          <ModelSelector
            models={models}
            selectedModel={selectedModel}
            onSelect={(id, provider) => {
              setSelectedModel(id);
              setSelectedProvider(provider);
            }}
          />
          <Button
            data-testid="chat-export"
            type="button"
            variant="outline"
            size="sm"
            onClick={exportChat}
            disabled={messages.length === 0}
            className="border-slate-700 text-slate-400 hover:bg-slate-800 disabled:opacity-40"
          >
            <Download className="h-4 w-4 mr-1.5" />
            Export
          </Button>
          <Button
            data-testid="chat-clear"
            type="button"
            variant="outline"
            size="sm"
            onClick={clearChat}
            disabled={messages.length === 0}
            className="border-slate-700 text-slate-400 hover:bg-slate-800 disabled:opacity-40"
          >
            <Trash2 className="h-4 w-4 mr-1.5" />
            Clear
          </Button>
        </div>
      </div>

      {/* Personality pills */}
      <div data-testid="personality-select" className="flex flex-wrap gap-2">
        <span className="text-xs text-slate-500 self-center mr-1">Personality</span>
        {PERSONALITIES.map((p) => (
          <button
            key={p.id}
            type="button"
            onClick={() => {
              setPersonalityId(p.id);
              setChatPrefs({ personalityId: p.id });
            }}
            className={cn(
              "px-2.5 py-1 rounded-full text-xs font-medium border transition-colors",
              personalityId === p.id
                ? `${PERSONALITY_COLORS[p.color]} text-white`
                : "border-slate-700 text-slate-400 hover:border-slate-600 hover:text-slate-300",
            )}
            title={p.description}
          >
            {p.name}
          </button>
        ))}
      </div>

      {/* Refinement toggle */}
      <label className="flex items-center gap-2 cursor-pointer w-fit">
        <input
          type="checkbox"
          checked={promptRefinement}
          onChange={(e) => {
            setPromptRefinement(e.target.checked);
            setChatPrefs({ promptRefinement: e.target.checked });
          }}
          className="rounded border-slate-600 bg-slate-900 text-emerald-500 focus:ring-emerald-500"
        />
        <span className="text-sm text-slate-400 flex items-center gap-1.5">
          <Sparkles className="h-3.5 w-3.5" />
          Refine prompt with LLM before sending
        </span>
      </label>

      {error && (
        <div className="text-red-400 text-sm bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
          {error}
        </div>
      )}

      <Card className="flex-1 border-slate-800 bg-slate-950/50 flex flex-col overflow-hidden min-h-0">
        <CardContent data-testid="chat-messages" className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 && (
            <div className="text-slate-500 text-sm text-center pt-4">
              <Bot className="w-10 h-10 mx-auto mb-2 opacity-20" />
              <p>Send a message{skillName ? ` (skill: ${skillName})` : ""}.</p>
              <p className="text-xs text-slate-600 mt-1">
                Personality:{" "}
                <span className={cn("font-medium", personalityColor)}>{personality.name}</span>
              </p>
              <div data-testid="example-prompts" className="mt-5 max-w-md mx-auto space-y-2.5">
                {EXAMPLE_PROMPTS.map((group) => (
                  <div key={group.group}>
                    <p className="text-[10px] uppercase tracking-wider text-slate-600 text-left mb-1 px-1">
                      {group.group}
                    </p>
                    <div className="flex flex-wrap gap-1.5 justify-center">
                      {group.prompts.map((p) => (
                        <button
                          key={p}
                          type="button"
                          onClick={() => setInput(p)}
                          className="text-xs px-2.5 py-1.5 rounded-lg border border-slate-700 bg-slate-800/50 hover:bg-slate-700/50 text-slate-400 hover:text-slate-200 transition-colors text-left"
                        >
                          {p}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={cn(
                "flex gap-3",
                msg.role === "assistant" && "border-l-4 pl-3 -ml-1 rounded-r",
                msg.role === "assistant" && personalityColor,
              )}
            >
              <div
                className={
                  msg.role === "user"
                    ? "h-8 w-8 rounded-full bg-slate-800 flex items-center justify-center border border-slate-700 flex-shrink-0"
                    : "h-8 w-8 rounded-full bg-blue-900/20 flex items-center justify-center border border-blue-800 flex-shrink-0"
                }
              >
                {msg.role === "user" ? (
                  <User className="h-4 w-4 text-slate-400" />
                ) : (
                  <Bot className="h-4 w-4 text-blue-400" />
                )}
              </div>
              <div className="flex-1 space-y-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-slate-400">
                    {msg.role === "user" ? "You" : "Assistant"}
                  </span>
                  <CopyButton text={msg.content} />
                  {msg.role === "assistant" && <SpeakButton text={msg.content} />}
                </div>
                {msg.role === "user" ? (
                  <p className="text-sm text-slate-300 bg-slate-900/50 p-3 rounded-md border border-slate-800 whitespace-pre-wrap break-words">
                    {msg.content}
                  </p>
                ) : (
                  <div className="text-sm text-slate-300 bg-slate-900/50 p-3 rounded-md border border-slate-800 prose prose-invert prose-sm max-w-none prose-p:my-1 prose-ul:my-1 prose-li:my-0">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
                )}
              </div>
            </div>
          ))}
          {(refining || loading) && (
            <div className="flex gap-3">
              <div className="h-8 w-8 rounded-full bg-blue-900/20 flex items-center justify-center border border-blue-800 flex-shrink-0">
                <Loader2 className="h-4 w-4 text-blue-400 animate-spin" />
              </div>
              <span className="text-sm text-slate-500">
                {refining ? "Refining prompt…" : "Generating…"}
              </span>
            </div>
          )}
          <div ref={scrollRef} />
        </CardContent>
        <MicButton input={input} setInput={setInput} />
        <form onSubmit={handleSubmit} className="p-4 border-t border-slate-800 bg-slate-900/30">
          <div className="flex gap-2">
            <input
              data-testid="chat-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={loading}
              className="flex-1 bg-slate-950 border border-slate-800 rounded-md px-4 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-emerald-500 resize-none disabled:opacity-50"
              placeholder="Message... (Enter to send)"
            />
            <Button
              data-testid="chat-send"
              type="submit"
              size="icon"
              className="bg-emerald-600 hover:bg-emerald-700 flex-shrink-0"
              disabled={loading}
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </form>
      </Card>
    </div>
  );
}
