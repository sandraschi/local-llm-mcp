import { Check, ChevronDown, Info, Search, Sparkles, Zap } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { ModelInfo } from "@/api/client";
import { cn } from "@/common/utils";
import { ModelIntelligenceCard } from "./ModelIntelligenceCard";

interface ModelSelectorProps {
  models: ModelInfo[];
  selectedModel: string;
  onSelect: (modelId: string, provider: string) => void;
  className?: string;
}

export function ModelSelector({ models, selectedModel, onSelect, className }: ModelSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [inspectingModel, setInspectingModel] = useState<ModelInfo | null>(null);
  const [filterCompatible, setFilterCompatible] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const currentModel = models.find((m) => (m.id ?? m.name) === selectedModel);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
        setInspectingModel(null);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const filteredModels = models.filter((m) => {
    const searchMatch =
      m.name.toLowerCase().includes(search.toLowerCase()) ||
      m.provider.toLowerCase().includes(search.toLowerCase());

    if (!searchMatch) return false;

    if (filterCompatible) {
      return m.hardware_compatibility === "READY" || m.hardware_compatibility === "TIGHT";
    }

    return true;
  });

  return (
    <div className={cn("relative", className)} ref={dropdownRef}>
      {/* Trigger */}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center justify-between gap-3 bg-slate-900/50 border border-slate-700/50 rounded-xl px-4 py-2.5 text-sm text-white hover:border-emerald-500/50 hover:bg-slate-800/50 transition-all min-w-[240px] shadow-lg shadow-black/20"
      >
        <div className="flex items-center gap-2.5 overflow-hidden">
          <div className="w-6 h-6 rounded-md bg-emerald-500/20 flex items-center justify-center flex-shrink-0">
            <Sparkles className="h-3.5 w-3.5 text-emerald-400" />
          </div>
          <div className="flex flex-col items-start truncate text-left">
            <span className="font-medium truncate leading-tight">
              {currentModel?.name ?? "Select Model"}
            </span>
            <span className="text-[10px] text-slate-500 uppercase tracking-wider">
              {currentModel?.provider ?? "Provider"}
            </span>
          </div>
        </div>
        <ChevronDown
          className={cn("h-4 w-4 text-slate-500 transition-transform", isOpen && "rotate-180")}
        />
      </button>

      {/* Dropdown Content */}
      {isOpen && (
        <div className="absolute top-full left-0 mt-2 z-[100] flex gap-2">
          {/* Main List */}
          <div className="w-80 glass-sidebar rounded-2xl border border-white/10 shadow-2xl overflow-hidden flex flex-col max-h-[480px]">
            {/* Search and Filters */}
            <div className="p-3 border-b border-white/5 bg-slate-950/20 space-y-2">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-slate-500" />
                <input
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search SOTA models..."
                  className="w-full bg-slate-900 border border-white/5 rounded-lg py-2 pl-9 pr-3 text-xs text-white focus:outline-none focus:ring-1 focus:ring-emerald-500/50"
                />
              </div>
              <div className="flex items-center justify-between px-1">
                <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">
                  Fit my hardware
                </span>
                <button
                  type="button"
                  onClick={() => setFilterCompatible(!filterCompatible)}
                  className={cn(
                    "w-7 h-4 rounded-full transition-colors relative",
                    filterCompatible ? "bg-emerald-500" : "bg-slate-700",
                  )}
                >
                  <div
                    className={cn(
                      "absolute top-0.5 w-3 h-3 bg-white rounded-full transition-all",
                      filterCompatible ? "left-3.5" : "left-0.5",
                    )}
                  />
                </button>
              </div>
            </div>

            {/* List */}
            <div className="flex-1 overflow-y-auto custom-scrollbar p-2">
              {filteredModels.length === 0 && (
                <div className="py-8 text-center text-slate-500 text-xs italic">
                  No models found matching your search.
                </div>
              )}
              {filteredModels.map((m) => {
                const isSelected = (m.id ?? m.name) === selectedModel;
                const intel = m.intelligence;
                const isLegacy = intel?.is_legacy;
                const comp = m.hardware_compatibility;

                return (
                  <div key={m.id ?? m.name} className="relative group">
                    <button
                      type="button"
                      onClick={() => {
                        onSelect(m.id ?? m.name, m.provider);
                        setIsOpen(false);
                        setInspectingModel(null);
                      }}
                      onMouseEnter={() => setInspectingModel(m)}
                      className={cn(
                        "w-full flex items-center justify-between p-2.5 rounded-xl transition-all mb-1",
                        isSelected
                          ? "bg-emerald-500/10 text-emerald-400"
                          : "hover:bg-white/[0.03] text-slate-300",
                        (isLegacy || comp === "OOM") && "opacity-50 grayscale hover:grayscale-0",
                        comp === "OOM" && "border border-rose-500/10",
                      )}
                    >
                      <div className="flex flex-col items-start overflow-hidden">
                        <div className="flex items-center gap-2">
                          <div
                            className={cn(
                              "w-1.5 h-1.5 rounded-full",
                              comp === "READY" && "bg-emerald-500",
                              comp === "TIGHT" && "bg-amber-500",
                              comp === "OOM" && "bg-rose-500 animate-pulse",
                              comp === "UNKNOWN" && "bg-slate-500",
                            )}
                            title={comp}
                          />
                          <span className="text-sm font-medium truncate">{m.name}</span>
                          {isSelected && <Check className="h-3.5 w-3.5" />}
                          {!isLegacy && intel?.release_date?.startsWith("2026") && (
                            <Zap className="h-3 w-3 text-amber-400" />
                          )}
                        </div>
                        <span className="text-[10px] text-slate-500 flex items-center gap-1.5 uppercase tracking-widest">
                          {m.provider}
                          {isLegacy && (
                            <span className="px-1.5 py-0.5 rounded-sm bg-slate-800 text-[8px] font-bold text-slate-400 border border-white/5">
                              LEGACY
                            </span>
                          )}
                        </span>
                      </div>
                      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button
                          type="button"
                          className="p-1.5 rounded-lg hover:bg-emerald-500/20 text-slate-500 hover:text-emerald-400"
                          onClick={(e) => {
                            e.stopPropagation();
                            setInspectingModel(m);
                          }}
                        >
                          <Info className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Intelligence Inspection Card (Appears on Hover/Click) */}
          {inspectingModel && (
            <div className="w-[420px] shadow-2xl animate-in fade-in slide-in-from-left-2 duration-200">
              <ModelIntelligenceCard model={inspectingModel} className="h-full" />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
