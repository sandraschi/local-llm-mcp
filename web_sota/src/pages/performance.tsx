import { Activity, Cpu, Flame, Gauge, HardDrive, Zap } from "lucide-react";
import { useEffect, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getTelemetry, type TelemetryResponse } from "@/api/telemetry";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

interface ChartData {
  time: string;
  load: number;
  power: number;
  temp: number;
}

export function Performance() {
  const [data, setData] = useState<TelemetryResponse | null>(null);
  const [history, setHistory] = useState<ChartData[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchTelemetry = async () => {
      try {
        const res = await getTelemetry();
        setData(res);
        setLoading(false);

        if (res.gpu.gpu) {
          const newEntry: ChartData = {
            time: new Date().toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
              second: "2-digit",
            }),
            load: res.gpu.gpu.utilization_gpu,
            power: res.gpu.gpu.power_draw,
            temp: res.gpu.gpu.temperature,
          };
          setHistory((prev) => [...prev.slice(-19), newEntry]);
        }
      } catch (err) {
        console.error("Telemetry fetch failed", err);
      }
    };

    fetchTelemetry();
    const interval = setInterval(fetchTelemetry, 2000);
    return () => clearInterval(interval);
  }, []);

  if (loading || !data?.gpu.gpu) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] space-y-4">
        <Activity className="h-12 w-12 text-emerald-500 animate-pulse" />
        <p className="text-slate-400 font-medium">Initializing Performance Foundry...</p>
      </div>
    );
  }

  const gpu = data.gpu.gpu;
  const vramPercent = (gpu.used_mb / gpu.total_mb) * 100;

  return (
    <div className="space-y-8 page-enter pb-12">
      <div className="flex flex-col gap-2">
        <h2 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
          <Gauge className="h-8 w-8 text-emerald-500" />
          GPU Performance Foundry
        </h2>
        <p className="text-slate-400 max-w-2xl">
          Real-time high-fidelity metrics for the local compute cluster. Monitoring{" "}
          <span className="text-emerald-400 font-mono">{gpu.name}</span>.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <Card className="glass-card overflow-hidden group">
          <CardHeader className="flex flex-row items-center justify-between pb-2 space-y-0">
            <CardTitle className="text-sm font-medium text-slate-400">GPU Core Load</CardTitle>
            <Cpu className="h-4 w-4 text-emerald-400 group-hover:scale-110 transition-transform" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold font-mono">{gpu.utilization_gpu}%</div>
            <Progress
              value={gpu.utilization_gpu}
              className="h-1 mt-3 bg-white/5"
              indicatorClassName="bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]"
            />
          </CardContent>
        </Card>

        <Card className="glass-card overflow-hidden group">
          <CardHeader className="flex flex-row items-center justify-between pb-2 space-y-0">
            <CardTitle className="text-sm font-medium text-slate-400">Power Draw</CardTitle>
            <Zap className="h-4 w-4 text-blue-400 group-hover:scale-110 transition-transform" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold font-mono">{gpu.power_draw}W</div>
            <p className="text-[10px] text-slate-500 mt-2 uppercase tracking-widest">
              Instantaneous Consumption
            </p>
          </CardContent>
        </Card>

        <Card className="glass-card overflow-hidden group">
          <CardHeader className="flex flex-row items-center justify-between pb-2 space-y-0">
            <CardTitle className="text-sm font-medium text-slate-400">Thermal Engine</CardTitle>
            <Flame className="h-4 w-4 text-orange-400 group-hover:scale-110 transition-transform" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold font-mono">{gpu.temperature}°C</div>
            <p className="text-[10px] text-slate-500 mt-2 uppercase tracking-widest">
              Core Temperature
            </p>
          </CardContent>
        </Card>

        <Card className="glass-card overflow-hidden group">
          <CardHeader className="flex flex-row items-center justify-between pb-2 space-y-0">
            <CardTitle className="text-sm font-medium text-slate-400">VRAM Allocation</CardTitle>
            <HardDrive className="h-4 w-4 text-purple-400 group-hover:scale-110 transition-transform" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold font-mono">{gpu.used_mb} MB</div>
            <p className="text-[10px] text-slate-500 mt-2 uppercase tracking-widest">
              {(gpu.used_mb / 1024).toFixed(1)}GB / {(gpu.total_mb / 1024).toFixed(1)}GB
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-7">
        <Card className="lg:col-span-5 glass-card">
          <CardHeader>
            <CardTitle className="text-lg font-semibold text-emerald-500/80 uppercase tracking-widest">
              Inference Force Trend
            </CardTitle>
          </CardHeader>
          <CardContent className="h-[400px] w-full pt-4">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={history}>
                <defs>
                  <linearGradient id="colorLoad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="colorPower" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
                <XAxis
                  dataKey="time"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: "#64748b", fontSize: 10 }}
                  minTickGap={30}
                />
                <YAxis axisLine={false} tickLine={false} tick={{ fill: "#64748b", fontSize: 10 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0f172a",
                    border: "1px solid #1e293b",
                    borderRadius: "12px",
                    fontSize: "12px",
                  }}
                  itemStyle={{ fontWeight: "bold" }}
                />
                <Area
                  type="monotone"
                  dataKey="load"
                  name="GPU Load %"
                  stroke="#10b981"
                  strokeWidth={2}
                  fillOpacity={1}
                  fill="url(#colorLoad)"
                />
                <Area
                  type="monotone"
                  dataKey="power"
                  name="Power (W)"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  fillOpacity={1}
                  fill="url(#colorPower)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="lg:col-span-2 glass-card">
          <CardHeader>
            <CardTitle className="text-lg font-semibold text-emerald-500/80 uppercase tracking-widest">
              Memory Reservoir
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6 pt-4">
            <div className="flex flex-col items-center justify-center py-6">
              <div className="relative w-40 h-40">
                <svg
                  className="w-full h-full transform -rotate-90"
                  aria-label="GPU utilization gauge"
                >
                  <circle
                    cx="80"
                    cy="80"
                    r="70"
                    stroke="currentColor"
                    strokeWidth="8"
                    fill="transparent"
                    className="text-white/5"
                  />
                  <circle
                    cx="80"
                    cy="80"
                    r="70"
                    stroke="currentColor"
                    strokeWidth="8"
                    fill="transparent"
                    strokeDasharray={440}
                    strokeDashoffset={440 - (440 * vramPercent) / 100}
                    className="text-emerald-500 transition-all duration-1000 ease-in-out"
                    strokeLinecap="round"
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className="text-2xl font-bold">{vramPercent.toFixed(0)}%</span>
                  <span className="text-[10px] uppercase text-slate-500 font-bold tracking-tighter">
                    Capacity Used
                  </span>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex justify-between items-center text-xs">
                <span className="text-slate-400">Total VRAM</span>
                <span className="text-slate-100 font-mono">
                  {(gpu.total_mb / 1024).toFixed(2)} GB
                </span>
              </div>
              <div className="flex justify-between items-center text-xs">
                <span className="text-slate-400">Available</span>
                <span className="text-emerald-400 font-mono">
                  {(gpu.free_mb / 1024).toFixed(2)} GB
                </span>
              </div>
              <div className="flex justify-between items-center text-xs">
                <span className="text-slate-400">In-Use</span>
                <span className="text-purple-400 font-mono">
                  {(gpu.used_mb / 1024).toFixed(2)} GB
                </span>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-white/[0.03] border border-white/[0.06] mt-4">
              <p className="text-[10px] text-slate-500 font-medium uppercase leading-tight">
                Recommended Config
              </p>
              <p className="mt-1 text-sm font-bold text-emerald-500 uppercase">
                {data.gpu.recommended_mode} MODE ACTIVE
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
