import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Send, Bot, User, Loader2, Trash2, Copy, Check, Sparkles, Mic, MicOff, Volume2 } from "lucide-react";
import { listModels, generate, type ModelInfo } from "@/api/client";
import { getDefaults } from "@/api/defaults";
import { getChatPrefs, setChatPrefs } from "@/api/chat-prefs";
import { PERSONALITIES, getPersonality } from "@/common/personalities";
import { speak, createSpeechRecognition, isTTSSupported, isSTTSupported } from "@/common/speech";
import { cn } from "@/common/utils";

const REFINE_PROMPT_PREFIX =
  "Rewrite the following into a single clear, self-contained prompt for an LLM. Preserve the user's intent. Output only the rewritten prompt, no commentary.\n\n";

type Message = { role: "user" | "assistant"; content: string };

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

function SpeakButton({ text }: { text: string }) {
  const [speaking, setSpeaking] = useState(false);
  if (!isTTSSupported()) return null;
  return (
    <button
      type="button"
      onClick={() => {
        if (speaking) {
          window.speechSynthesis.cancel();
          setSpeaking(false);
          return;
        }
        setSpeaking(true);
        speak(text, () => setSpeaking(false));
      }}
      className={cn(
        "p-1.5 rounded transition-colors",
        speaking ? "text-amber-400 bg-amber-500/20" : "text-slate-400 hover:text-white hover:bg-white/10"
      )}
      title={speaking ? "Stop" : "Speak"}
    >
      <Volume2 className="h-3.5 w-3.5" />
    </button>
  );
}

export function Chat() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedProvider, setSelectedProvider] = useState<string | undefined>();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [refining, setRefining] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [listening, setListening] = useState(false);
  const [interimTranscript, setInterimTranscript] = useState("");
  const recognitionRef = useRef<ReturnType<typeof createSpeechRecognition> | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const prefs = getChatPrefs();
  const [personalityId, setPersonalityId] = useState(prefs.personalityId);
  const [promptRefinement, setPromptRefinement] = useState(prefs.promptRefinement);

  const personality = getPersonality(personalityId) ?? getPersonality("neutral")!;
  const personalityColor = PERSONALITY_COLORS[personality.color] ?? PERSONALITY_COLORS.slate;

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
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    if (!isSTTSupported()) return;
    recognitionRef.current = createSpeechRecognition(
      (transcript, isFinal) => {
        if (isFinal) {
          setInput((prev) => (prev ? prev + " " + transcript : transcript));
          setInterimTranscript("");
        } else {
          setInterimTranscript(transcript);
        }
      },
      () => setListening(false)
    );
    return () => {
      recognitionRef.current?.stop();
    };
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
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setLoading(true);

    let promptToSend = text;
    if (promptRefinement) {
      setRefining(true);
      try {
        const refined = await generate(
          REFINE_PROMPT_PREFIX + text,
          selectedModel,
          selectedProvider
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
      setMessages((prev) => [...prev, { role: "assistant", content: res.text }]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
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
  }

  function toggleMic() {
    if (!recognitionRef.current) return;
    if (listening) {
      recognitionRef.current.stop();
    } else {
      setInterimTranscript("");
      recognitionRef.current.start();
      setListening(true);
    }
  }

  return (
    <div className="flex h-[calc(100vh-8rem)] flex-col space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-white">Chat</h2>
          <p className="text-slate-400">Personality, refinement, and model</p>
        </div>
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-2">
            <label className="text-sm text-slate-400">Model</label>
            <select
              value={selectedModel}
              onChange={(e) => {
                const m = models.find((x) => (x.id ?? x.name) === e.target.value);
                setSelectedModel(e.target.value);
                if (m) setSelectedProvider(m.provider);
              }}
              className="bg-slate-900 border border-slate-700 rounded-md px-3 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-emerald-500"
            >
              {models.length === 0 && <option value="">No models</option>}
              {models.map((m) => (
                <option key={m.id ?? m.name} value={m.id ?? m.name}>
                  {m.name} ({m.provider})
                </option>
              ))}
            </select>
          </div>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={clearChat}
            className="border-slate-700 text-slate-400 hover:bg-slate-800"
          >
            <Trash2 className="h-4 w-4 mr-1.5" />
            Clear
          </Button>
        </div>
      </div>

      {/* Personality pills */}
      <div className="flex flex-wrap gap-2">
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
                ? PERSONALITY_COLORS[p.color] + " text-white"
                : "border-slate-700 text-slate-400 hover:border-slate-600 hover:text-slate-300"
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
        <CardContent className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 && (
            <p className="text-slate-500 text-sm">
              Send a message. Personality: <span className={cn("font-medium", personalityColor)}>{personality.name}</span> — {personality.description}
            </p>
          )}
          {messages.map((msg, i) => (
            <div
              key={i}
              className={cn(
                "flex gap-3",
                msg.role === "assistant" && "border-l-4 pl-3 -ml-1 rounded-r",
                msg.role === "assistant" && personalityColor
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
        <form onSubmit={handleSubmit} className="p-4 border-t border-slate-800 bg-slate-900/30">
          {interimTranscript && (
            <p className="text-xs text-slate-500 mb-2 italic">&ldquo;{interimTranscript}&rdquo;</p>
          )}
          <div className="flex gap-2">
            {isSTTSupported() && (
              <Button
                type="button"
                size="icon"
                variant="outline"
                onClick={toggleMic}
                className={cn(
                  "border-slate-700 flex-shrink-0",
                  listening ? "text-red-400 border-red-500/50 bg-red-500/10 animate-pulse" : "text-slate-400 hover:bg-slate-800"
                )}
                title={listening ? "Stop listening" : "Voice input"}
              >
                {listening ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
              </Button>
            )}
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={loading}
              className="flex-1 bg-slate-950 border border-slate-800 rounded-md px-4 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-emerald-500 resize-none disabled:opacity-50"
              placeholder="Message... (Enter to send)"
            />
            <Button type="submit" size="icon" className="bg-emerald-600 hover:bg-emerald-700 flex-shrink-0" disabled={loading}>
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </form>
      </Card>
    </div>
  );
}
