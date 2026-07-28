export interface GPUStats {
  name: string;
  total_mb: number;
  free_mb: number;
  used_mb: number;
  total_gb: number;
  free_gb: number;
  temperature: number;
  utilization_gpu: number;
  utilization_mem: number;
  power_draw: number;
  clock_speed: number;
}

export interface TelemetryResponse {
  gpu: {
    available: boolean;
    gpu: GPUStats | null;
    recommended_mode: string;
    timestamp: string;
  };
  system: {
    status: string;
  };
}

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:10833";

export async function getTelemetry(): Promise<TelemetryResponse> {
  const r = await fetch(`${API_BASE}/api/v1/telemetry/`);
  if (!r.ok) throw new Error(`Telemetry failed: ${r.status}`);
  return r.json();
}
