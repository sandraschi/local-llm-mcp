import { Activity, Clock, Cpu, Database, ShieldCheck, Zap } from "lucide-react";
import { cn } from "@/common/utils";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function Analytics() {
  const stats = [
    { label: "System Uptime", value: "14d 6h 22m", icon: Clock, color: "text-blue-400" },
    { label: "Engine Status", value: "Healthy", icon: ShieldCheck, color: "text-emerald-400" },
    { label: "Model Context", value: "128k", icon: Database, color: "text-violet-400" },
    { label: "Avg Latency", value: "12ms", icon: Zap, color: "text-amber-400" },
  ];

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
          <Activity className="h-8 w-8 text-blue-500" />
          Engine Analytics
        </h2>
        <p className="text-slate-400 mt-2">
          Real-time telemetry and performance orchestration metrics for the LLM infrastructure.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((s) => (
          <Card key={s.label} className="border-white/5 bg-white/[0.02] backdrop-blur-md">
            <CardHeader className="flex flex-row items-center justify-between pb-2 space-y-0">
              <CardTitle className="text-xs font-medium text-slate-500 uppercase tracking-wider">
                {s.label}
              </CardTitle>
              <s.icon className={`h-4 w-4 ${s.color}`} />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">{s.value}</div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="border-white/5 bg-white/[0.02] backdrop-blur-md">
          <CardHeader>
            <CardTitle className="text-lg text-white flex items-center gap-2">
              <Cpu className="h-5 w-5 text-emerald-500" />
              Resource Allocation
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {[
              { label: "GPU VRAM Utilization", value: 68, color: "bg-emerald-500" },
              { label: "System RAM", value: 42, color: "bg-blue-500" },
              { label: "NPU Load", value: 12, color: "bg-violet-500" },
            ].map((bar) => (
              <div key={bar.label} className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400">{bar.label}</span>
                  <span className="text-slate-200">{bar.value}%</span>
                </div>
                <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${bar.color} transition-all duration-1000`}
                    style={{ width: `${bar.value}%` }}
                  />
                </div>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card className="border-white/5 bg-white/[0.02] backdrop-blur-md">
          <CardHeader>
            <CardTitle className="text-lg text-white flex items-center gap-2">
              <ShieldCheck className="h-5 w-5 text-blue-500" />
              Recent Security Events
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {[
                { time: "2m ago", event: "Ollama Provider Heartbeat", status: "ok" },
                { time: "15m ago", event: "New Model Registered: llama3.1", status: "info" },
                { time: "1h ago", event: "API Config Patched via WebUI", status: "warn" },
                { time: "2h ago", event: "vLLM Instance Re-optimized", status: "ok" },
              ].map((log, _i) => (
                <div
                  key={log.event}
                  className="flex items-center justify-between text-xs border-b border-white/5 pb-2 last:border-0"
                >
                  <span className="text-slate-500 font-mono">{log.time}</span>
                  <span className="text-slate-300 flex-1 ml-4">{log.event}</span>
                  <span
                    className={cn(
                      "px-1.5 py-0.5 rounded text-[10px] font-bold uppercase",
                      log.status === "ok"
                        ? "text-emerald-500"
                        : log.status === "warn"
                          ? "text-amber-500"
                          : "text-blue-500",
                    )}
                  >
                    {log.status}
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
