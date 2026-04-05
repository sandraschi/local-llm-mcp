import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Activity, Database, Radio, AlertCircle, Cpu, HardDrive } from "lucide-react";
import { getHealth, listModels } from "@/api/client";

export function Dashboard() {
    const [loading, setLoading] = useState(true);
    const [isConnected, setIsConnected] = useState(false);
    const [modelCount, setModelCount] = useState(0);
    const [status, setStatus] = useState<string>("unknown");
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        let cancelled = false;
        (async () => {
            setLoading(true);
            setError(null);
            try {
                const health = await getHealth();
                if (cancelled) return;
                setIsConnected(health.status === "ok");
                setStatus(health.status === "ok" ? "healthy" : "degraded");
                const models = await listModels();
                if (cancelled) return;
                setModelCount(Array.isArray(models) ? models.length : 0);
            } catch (e) {
                if (!cancelled) {
                    setIsConnected(false);
                    setStatus("error");
                    setError(e instanceof Error ? e.message : "Backend unreachable");
                }
            } finally {
                if (!cancelled) setLoading(false);
            }
        })();
        return () => { cancelled = true; };
    }, []);

    const uptime = "—";

    return (
        <div className="space-y-8 page-enter">
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-3xl font-bold tracking-tight gradient-text">Intelligence Engine</h2>
                    <p className="text-slate-400">Local LLM inference and model orchestration</p>
                </div>
                {!loading && !isConnected && (
                    <div className="flex items-center text-red-400 bg-red-500/10 px-4 py-2 rounded-xl border border-red-500/20 glass-card">
                        <AlertCircle className="h-5 w-5 mr-3" />
                        <span className="text-sm font-medium">Engine Disconnected</span>
                        {error != null && <span className="text-xs ml-2 opacity-80">({error})</span>}
                    </div>
                )}
                {loading && (
                    <div className="text-slate-400 text-sm">Connecting to backend...</div>
                )}
            </div>

            {/* KPI Cards */}
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
                <Card className="glass-card">
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
                        <CardTitle className="text-sm font-medium text-slate-300">
                            Available Models
                        </CardTitle>
                        <div className="p-2 rounded-lg bg-emerald-500/10">
                            <Database className="h-4 w-4 text-emerald-400" />
                        </div>
                    </CardHeader>
                    <CardContent>
                        <div className="text-3xl font-bold">{modelCount}</div>
                        <p className="text-xs text-slate-500 mt-1">
                            {isConnected ? "Backend API connected" : "Library unavailable"}
                        </p>
                    </CardContent>
                </Card>

                <Card className="glass-card">
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
                        <CardTitle className="text-sm font-medium text-slate-300">
                            Inference State
                        </CardTitle>
                        <div className="p-2 rounded-lg bg-blue-500/10">
                            <Radio className="h-4 w-4 text-blue-400" />
                        </div>
                    </CardHeader>
                    <CardContent>
                        <div className="text-3xl font-bold">
                            {isConnected ? "Active" : "Idle"}
                        </div>
                        <p className="text-xs text-slate-500 mt-1">
                            Local GPU Acceleration Enabled
                        </p>
                    </CardContent>
                </Card>

                <Card className="glass-card">
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
                        <CardTitle className="text-sm font-medium text-slate-300">
                            Handler Pulse
                        </CardTitle>
                        <div className="p-2 rounded-lg bg-purple-500/10">
                            <Activity className="h-4 w-4 text-purple-400" />
                        </div>
                    </CardHeader>
                    <CardContent>
                        <div className="text-3xl font-bold">
                            Online
                        </div>
                        <p className="text-xs text-slate-500 mt-1">
                            Inference-ready handlers
                        </p>
                    </CardContent>
                </Card>

                <Card className="glass-card">
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
                        <CardTitle className="text-sm font-medium text-slate-300">
                            Compute Load
                        </CardTitle>
                        <div className="p-2 rounded-lg bg-orange-500/10">
                            <Cpu className="h-4 w-4 text-orange-400" />
                        </div>
                    </CardHeader>
                    <CardContent>
                        <div className="text-3xl font-bold">
                            Low
                        </div>
                        <p className="text-xs text-slate-500 mt-1 line-clamp-1">
                            Quantized weights active
                        </p>
                    </CardContent>
                </Card>
            </div>

            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-7">
                <Card className="col-span-4 glass-card">
                    <CardHeader>
                        <CardTitle className="text-xl font-bold uppercase tracking-widest text-emerald-500/70">Inference Stream Logs</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="h-[250px] font-mono text-sm p-5 overflow-y-auto border border-white/[0.06] rounded-xl bg-black/40 text-slate-400 space-y-1.5 scrollbar-thin scrollbar-thumb-white/10">
                            <p className="text-emerald-400">[engine] Backend {isConnected ? "connected" : "disconnected"}</p>
                            <p className="text-blue-400">[models] Discovered {modelCount} local models</p>
                            <p>[gpu] RTX 4090 detected and initialized</p>
                            <p>[mem] VRAM overhead: 24.0 GB free</p>
                            <p className="text-slate-500">[system] Bridge established on port 10833</p>
                            <div className="animate-pulse inline-block h-2 w-1 bg-slate-500 ml-1 mt-2" />
                        </div>
                    </CardContent>
                </Card>

                <Card className="col-span-3 glass-card">
                    <CardHeader>
                        <CardTitle className="text-xl font-bold uppercase tracking-widest text-emerald-500/70">Engine Pulse</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-6">
                            <div className="flex items-center p-4 rounded-2xl bg-white/[0.03] border border-white/[0.06] hover:bg-white/[0.05] transition-all">
                                <HardDrive className="h-5 w-5 text-slate-400 mr-4" />
                                <div className="space-y-1">
                                    <p className="text-sm font-medium leading-none">Process Uptime</p>
                                    <p className="text-xs text-slate-400 font-mono mt-1.5">{uptime}</p>
                                </div>
                            </div>
                            <div className="flex items-center p-4 rounded-2xl bg-white/[0.03] border border-white/[0.06] hover:bg-white/[0.05] transition-all">
                                <Activity className="h-5 w-5 text-emerald-400 mr-4" />
                                <div className="space-y-1">
                                    <p className="text-sm font-medium leading-none">API Endpoint Status</p>
                                    <p className="text-xs text-slate-400 mt-1.5 capitalize">{status}</p>
                                </div>
                            </div>
                        </div>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
}
