import React from "react";
import { HelpCircle, Terminal, Cpu, Zap, Info, ShieldCheck, Globe } from "lucide-react";

export const Help: React.FC = () => {
    return (
        <div className="p-8 max-w-7xl mx-auto animate-in fade-in duration-700">
            <div className="mb-8 flex items-center gap-4">
                <div className="p-3 bg-blue-500/10 rounded-2xl border border-blue-500/20">
                    <HelpCircle className="w-8 h-8 text-blue-400" />
                </div>
                <div>
                    <h1 className="text-4xl font-bold text-white tracking-tight">Fleet Help & Documentation</h1>
                    <p className="text-gray-400 mt-1">Master your local inference infrastructure and agentic workflows.</p>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {/* Qwen 3.6 Spotlight */}
                <div className="col-span-1 lg:col-span-2 p-6 rounded-3xl bg-gradient-to-br from-indigo-500/10 via-transparent to-purple-500/10 border border-white/10 backdrop-blur-xl relative overflow-hidden group">
                    <div className="absolute top-0 right-0 p-8 opacity-10 group-hover:opacity-20 transition-opacity">
                        <Zap className="w-32 h-32 text-indigo-400" />
                    </div>
                    <div className="relative z-10">
                        <div className="flex items-center gap-2 mb-4">
                            <span className="px-3 py-1 bg-indigo-500/20 text-indigo-300 text-xs font-bold rounded-full border border-indigo-500/30 uppercase tracking-widest">Flagship Model</span>
                            <span className="px-3 py-1 bg-purple-500/20 text-purple-300 text-xs font-bold rounded-full border border-purple-500/30 uppercase tracking-widest">SOTA 2026</span>
                        </div>
                        <h2 className="text-3xl font-bold text-white mb-2">Qwen 3.6-35B-A3B Integration</h2>
                        <p className="text-gray-300 leading-relaxed mb-6 max-w-2xl">
                            The current standard for local **Agentic Coding**. This sparse Mixture-of-Experts (MoE) model activates only 3 billion parameters per token, providing unmatched speed for complex multi-file reasoning tasks.
                        </p>
                        <div className="flex flex-wrap gap-4">
                            <div className="flex items-center gap-2 px-4 py-2 bg-white/5 rounded-xl border border-white/10 text-sm font-semibold text-gray-200">
                                <Terminal className="w-4 h-4" />
                                <code>ollama run qwen3.6:35b-a3b</code>
                            </div>
                        </div>
                    </div>
                </div>

                {/* SOTA Metrics Card */}
                <div className="p-6 rounded-3xl bg-white/5 border border-white/10 backdrop-blur-xl flex flex-col justify-between">
                    <div>
                        <div className="p-3 bg-emerald-500/10 rounded-2xl border border-emerald-500/20 w-fit mb-4">
                            <ShieldCheck className="w-6 h-6 text-emerald-400" />
                        </div>
                        <h3 className="text-xl font-bold text-white">Security & Privacy</h3>
                        <p className="text-gray-400 mt-2 text-sm leading-relaxed">
                            Your agentic fleet operates on a **Zero-Telemetry** principle. All inference data remains within your local network, processed by Ollama, LM Studio, or vLLM.
                        </p>
                    </div>
                    <div className="mt-6 pt-4 border-t border-white/5">
                        <div className="flex items-center justify-between text-xs text-gray-500">
                            <span>Encryption Status</span>
                            <span className="text-emerald-400 font-bold tracking-widest">ACTIVE</span>
                        </div>
                    </div>
                </div>

                {/* Discovery Guide */}
                <div className="p-6 rounded-3xl bg-white/5 border border-white/10 backdrop-blur-xl group hover:border-blue-500/30 transition-all">
                    <div className="p-3 bg-blue-500/10 rounded-2xl border border-blue-500/20 w-fit mb-4 group-hover:bg-blue-500/20 transition-colors">
                        <Globe className="w-6 h-6 text-blue-400" />
                    </div>
                    <h3 className="text-xl font-bold text-white">Automated Discovery</h3>
                    <p className="text-blue-400 text-xs font-bold mt-1 uppercase tracking-tighter">Preferred (SOTA)</p>
                    <p className="text-gray-400 mt-3 text-sm leading-relaxed">
                        The fleet server heartbeats every 5 seconds to scan local ports (**11434**, **1234**, **8000**). If a model is running, it joins the federation automatically.
                    </p>
                    <ul className="mt-4 space-y-2">
                        <li className="flex items-center gap-2 text-xs text-gray-300">
                            <div className="w-1.5 h-1.5 rounded-full bg-blue-400" />
                            Zero-Config Elicitation
                        </li>
                        <li className="flex items-center gap-2 text-xs text-gray-300">
                            <div className="w-1.5 h-1.5 rounded-full bg-blue-400" />
                            Live Health Telemetry
                        </li>
                    </ul>
                </div>

                {/* Hardware Recommendations */}
                <div className="p-6 rounded-3xl bg-white/5 border border-white/10 backdrop-blur-xl">
                    <div className="p-3 bg-orange-500/10 rounded-2xl border border-orange-500/20 w-fit mb-4">
                        <Cpu className="w-6 h-6 text-orange-400" />
                    </div>
                    <h3 className="text-xl font-bold text-white">Hardware Tiers</h3>
                    <div className="mt-4 space-y-4">
                        <div>
                            <div className="flex items-center justify-between text-xs mb-1">
                                <span className="text-gray-300">VRAM Budget (Low)</span>
                                <span className="text-gray-500 text-[10px]">8-12GB</span>
                            </div>
                            <div className="w-full bg-white/5 rounded-full h-1 overflow-hidden">
                                <div className="bg-orange-400 h-full w-[30%]" />
                            </div>
                        </div>
                        <div>
                            <div className="flex items-center justify-between text-xs mb-1">
                                <span className="text-gray-300">VRAM Budget (Mid)</span>
                                <span className="text-gray-500 text-[10px]">16-24GB</span>
                            </div>
                            <div className="w-full bg-white/5 rounded-full h-1 overflow-hidden">
                                <div className="bg-orange-400 h-full w-[60%]" />
                            </div>
                        </div>
                        <div>
                            <div className="flex items-center justify-between text-xs mb-1">
                                <span className="text-gray-300">VRAM Budget (SOTA)</span>
                                <span className="text-gray-500 text-[10px]">32GB+</span>
                            </div>
                            <div className="w-full bg-white/5 rounded-full h-1 overflow-hidden">
                                <div className="bg-orange-400 h-full w-[100%]" />
                            </div>
                        </div>
                    </div>
                    <p className="text-[10px] text-gray-500 mt-4 italic text-center">
                        Optimization: FlashAttention 3 + Unsloth dynamic quantization.
                    </p>
                </div>

                {/* Legacy Setup Card */}
                <div className="p-6 rounded-3xl bg-white/5 border border-white/10 backdrop-blur-xl border-dashed opacity-70">
                    <div className="p-3 bg-gray-500/10 rounded-2xl border border-gray-500/20 w-fit mb-4">
                        <Info className="w-6 h-6 text-gray-400" />
                    </div>
                    <h3 className="text-xl font-bold text-white">Manual Setup</h3>
                    <p className="text-gray-500 text-xs font-bold mt-1 uppercase tracking-tighter">Legacy Pattern</p>
                    <p className="text-gray-500 mt-3 text-sm leading-relaxed lowercase italic">
                        Manual port mapping and static model manifests. Use only if discovery fails or if tunneling through an external VPN.
                    </p>
                    <div className="mt-4 flex gap-2">
                        <span className="px-2 py-1 bg-gray-500/10 text-gray-500 text-[10px] rounded border border-gray-500/20">.env override</span>
                        <span className="px-2 py-1 bg-gray-500/10 text-gray-500 text-[10px] rounded border border-gray-500/20">Model Hub</span>
                    </div>
                </div>
            </div>

            <div className="mt-12 p-8 rounded-[40px] bg-gradient-to-r from-blue-600/20 to-indigo-600/20 border border-blue-500/30 flex flex-col md:flex-row items-center justify-between gap-8">
                <div>
                    <h2 className="text-3xl font-bold text-white mb-2">Need Deep Support?</h2>
                    <p className="text-gray-300">Check the central fleet documentation for advanced architectural patterns.</p>
                </div>
                <a 
                    href="https://github.com/sandraschi/mcp-central-docs" 
                    target="_blank" 
                    rel="noreferrer"
                    className="px-8 py-4 bg-white text-blue-900 font-bold rounded-2xl hover:bg-blue-50 hover:scale-105 active:scale-95 transition-all whitespace-nowrap"
                >
                    Visit Central Docs
                </a>
            </div>
        </div>
    );
};
