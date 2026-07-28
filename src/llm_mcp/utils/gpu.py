"""
local-llm-mcp - GPU Detection
GPU and VRAM detection utilities ported from SongGeneration-Studio.
"""

import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("llm_mcp.utils.gpu")


def get_gpu_info() -> dict[str, Any]:
    """Detect GPU and available VRAM using nvidia-smi with advanced telemetry."""
    try:
        # Query more detailed fields: temperature, utilization, power, and clock
        query_fields = [
            "name",
            "memory.total",
            "memory.free",
            "memory.used",
            "temperature.gpu",
            "utilization.gpu",
            "utilization.memory",
            "power.draw",
            "clocks.gr",
        ]
        result = subprocess.run(
            [
                "C:\\Windows\\System32\\nvidia-smi.exe",
                f"--query-gpu={','.join(query_fields)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            shell=False,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= len(query_fields):
                    # Clean up numeric values that might have extra text or be 'N/A'
                    def clean_val(val: str) -> float:
                        if "N/A" in val or not val:
                            return 0.0
                        try:
                            return float(val.split()[0])
                        except (ValueError, IndexError):
                            return 0.0

                    gpus.append(
                        {
                            "name": parts[0],
                            "total_mb": int(clean_val(parts[1])),
                            "free_mb": int(clean_val(parts[2])),
                            "used_mb": int(clean_val(parts[3])),
                            "total_gb": round(clean_val(parts[1]) / 1024, 1),
                            "free_gb": round(clean_val(parts[2]) / 1024, 1),
                            "temperature": clean_val(parts[4]),
                            "utilization_gpu": clean_val(parts[5]),
                            "utilization_mem": clean_val(parts[6]),
                            "power_draw": clean_val(parts[7]),
                            "clock_speed": clean_val(parts[8]),
                        }
                    )
            if gpus:
                gpu = gpus[0]  # Primary GPU
                # Recommended mode based on VRAM (targeting local hardware like 3090/4090)
                if gpu["free_gb"] >= 23:
                    recommended = "full"
                elif gpu["free_gb"] >= 12:
                    recommended = "balanced"
                else:
                    recommended = "low"

                return {
                    "available": True,
                    "gpu": gpu,
                    "recommended_mode": recommended,
                    "can_run_full": gpu["free_gb"] >= 23,
                    "can_run_balanced": gpu["free_gb"] >= 12,
                    "can_run_low": gpu["free_gb"] >= 6,
                    "timestamp": datetime.utcnow().isoformat(),
                }
    except Exception as e:
        logger.error(f"GPU Telemetry error: {e}")

    return {
        "available": False,
        "gpu": None,
        "recommended_mode": "low",
        "can_run_full": False,
        "can_run_balanced": False,
        "can_run_low": False,
    }


# ============================================================================
# Audio Duration Helper (Kept for compatibility if needed, though less relevant for LLM)
# ============================================================================


def get_audio_duration(audio_path: Path) -> float | None:
    """Get audio duration in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception as e:
        logger.error(f"Failed to get duration for {audio_path}: {e}")
    return None


# Global GPU info - initialized on import
gpu_info = get_gpu_info()


def log_gpu_info():
    """Log GPU detection results"""
    if gpu_info["available"]:
        logger.info(f"Connected: {gpu_info['gpu']['name']}")
        logger.info(f"VRAM: {gpu_info['gpu']['free_gb']}GB free / {gpu_info['gpu']['total_gb']}GB total")
        logger.info(f"Temp: {gpu_info['gpu']['temperature']}°C | Power: {gpu_info['gpu']['power_draw']}W")
        logger.info(f"Recommended mode: {gpu_info['recommended_mode']}")
    else:
        logger.warning("No NVIDIA GPU detected or nvidia-smi not available")


def refresh_gpu_info() -> dict[str, Any]:
    """Refresh GPU info and return updated data"""
    global gpu_info
    gpu_info = get_gpu_info()
    return gpu_info
