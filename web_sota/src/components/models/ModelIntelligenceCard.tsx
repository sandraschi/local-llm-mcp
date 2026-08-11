import { Brain, Calendar, ExternalLink, Info, ShieldAlert, Sparkles, Zap } from "lucide-react";
import type { ModelInfo } from "@/api/client";
import { cn } from "@/common/utils";

interface ModelIntelligenceCardProps {
  model: ModelInfo;
  className?: string;
}

export function ModelIntelligenceCard({ model, className }: ModelIntelligenceCardProps) {
  const intel = model.intelligence;

  if (!intel) {
    return (
      <div
        className={cn(
          "glass-card p-6 flex flex-col items-center justify-center text-center space-y-3",
          className,
        )}
      >
        <Info className="h-8 w-8 text-slate-500" />
        <p className="text-slate-400 text-sm">
          No detailed intelligence available for this model yet.
        </p>
      </div>
    );
  }

  return (
    <div className={cn("glass-card overflow-hidden flex flex-col", className)}>
      {/* Header */}
      <div className="bg-emerald-600/10 border-b border-emerald-500/20 p-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-emerald-500/20 flex items-center justify-center emerald-glow">
            <Brain className="h-5 w-5 text-emerald-400" />
          </div>
          <div>
            <h3 className="font-bold text-slate-100">{model.name}</h3>
            <p className="text-xs text-slate-400">{intel.developer ?? "Unknown Developer"}</p>
          </div>
        </div>
        {intel.is_legacy && (
          <div className="flex items-center gap-1.5 px-2 py-1 rounded-full bg-amber-500/10 border border-amber-500/30 text-[10px] font-bold text-amber-500 uppercase tracking-wider">
            <ShieldAlert className="h-3 w-3" />
            Legacy
          </div>
        )}
      </div>

      <div className="p-6 space-y-6 flex-1">
        {/* Intro Date & HF Link */}
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center gap-2 text-slate-400">
            <Calendar className="h-3.5 w-3.5" />
            <span>Released: {intel.release_date ?? "Unknown"}</span>
          </div>
          {intel.model_card_url && (
            <a
              href={intel.model_card_url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 text-emerald-400 hover:text-emerald-300 transition-colors"
            >
              Model Card
              <ExternalLink className="h-3 w-3" />
            </a>
          )}
        </div>

        {/* Best For */}
        {intel.best_for && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm font-semibold text-slate-200">
              <Sparkles className="h-4 w-4 text-emerald-400" />
              Best For
            </div>
            <p className="text-sm text-slate-400 leading-relaxed italic border-l-2 border-emerald-500/30 pl-3">
              "{intel.best_for}"
            </p>
          </div>
        )}

        {/* Strengths & Weaknesses */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="text-[11px] font-bold text-emerald-500 uppercase tracking-widest pl-1">
              Strengths
            </div>
            <ul className="space-y-1.5">
              {intel.strengths.map((s) => (
                <li key={s} className="text-xs text-slate-300 flex items-start gap-2">
                  <div className="w-1 h-1 rounded-full bg-emerald-500 mt-1.5" />
                  {s}
                </li>
              ))}
              {intel.strengths.length === 0 && (
                <li className="text-xs text-slate-500 italic">No strengths listed.</li>
              )}
            </ul>
          </div>
          <div className="space-y-2">
            <div className="text-[11px] font-bold text-rose-500 uppercase tracking-widest pl-1">
              Weaknesses
            </div>
            <ul className="space-y-1.5">
              {intel.weaknesses.map((w) => (
                <li key={w} className="text-xs text-slate-300 flex items-start gap-2">
                  <div className="w-1 h-1 rounded-full bg-rose-500 mt-1.5" />
                  {w}
                </li>
              ))}
              {intel.weaknesses.length === 0 && (
                <li className="text-xs text-slate-500 italic">No weaknesses listed.</li>
              )}
            </ul>
          </div>
        </div>

        {/* Hardware Compatibility */}
        <div className="mt-4 pt-4 border-t border-white/5 space-y-3">
          <div className="flex items-center justify-between text-[11px] font-bold uppercase tracking-widest">
            <div className="flex items-center gap-2 text-blue-400">
              <Zap className="h-3 w-3" />
              Hardware Compatibility
            </div>
            {model.hardware_compatibility && (
              <span
                className={cn(
                  "px-1.5 py-0.5 rounded text-[9px]",
                  model.hardware_compatibility === "READY" &&
                    "bg-emerald-500/10 text-emerald-500 border border-emerald-500/20",
                  model.hardware_compatibility === "TIGHT" &&
                    "bg-amber-500/10 text-amber-500 border border-amber-500/20",
                  model.hardware_compatibility === "OOM" &&
                    "bg-rose-500/10 text-rose-500 border border-rose-500/20",
                  model.hardware_compatibility === "UNKNOWN" &&
                    "bg-slate-500/10 text-slate-500 border border-slate-500/20",
                )}
              >
                {model.hardware_compatibility}
              </span>
            )}
          </div>

          <div className="space-y-1.5">
            <div className="flex justify-between text-[10px] text-slate-500">
              <span>Estimated VRAM Required</span>
              <span>{intel.vram_required_gb ? `${intel.vram_required_gb}GB` : "N/A"}</span>
            </div>
            <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
              <div
                className={cn(
                  "h-full transition-all duration-500",
                  model.hardware_compatibility === "READY" &&
                    "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]",
                  model.hardware_compatibility === "TIGHT" && "bg-amber-500",
                  model.hardware_compatibility === "OOM" && "bg-rose-500",
                  !model.hardware_compatibility && "bg-slate-600",
                )}
                style={{
                  width:
                    model.hardware_compatibility === "READY" ||
                    model.hardware_compatibility === "TIGHT"
                      ? `${Math.min(100, (intel.vram_required_gb || 0) * 2)}%` // Just a visual representation
                      : "100%",
                }}
              />
            </div>
          </div>

          {intel.quantization_info && (
            <p className="text-xs text-slate-400 leading-relaxed italic">
              {intel.quantization_info}
            </p>
          )}
        </div>
      </div>

      {/* Visual Accent */}
      <div className="h-1 w-full bg-gradient-to-r from-emerald-500 via-blue-500 to-emerald-500 opacity-30" />
    </div>
  );
}
