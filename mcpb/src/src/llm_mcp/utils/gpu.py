"""
local-llm-mcp - GPU Detection
GPU and VRAM detection utilities ported from SongGeneration-Studio.
"""

import subprocess
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger("llm_mcp.utils.gpu")


def get_gpu_info() -> Dict[str, Any]:
    """Detect GPU and available VRAM using nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpus.append(
                        {
                            "name": parts[0],
                            "total_mb": int(parts[1]),
                            "free_mb": int(parts[2]),
                            "used_mb": int(parts[3]),
                            "total_gb": round(int(parts[1]) / 1024, 1),
                            "free_gb": round(int(parts[2]) / 1024, 1),
                        }
                    )
            if gpus:
                gpu = gpus[0]  # Primary GPU
                if gpu["free_gb"] >= 24:
                    recommended = "full"
                else:
                    recommended = "low"
                return {
                    "available": True,
                    "gpu": gpu,
                    "recommended_mode": recommended,
                    "can_run_full": gpu["free_gb"] >= 24,
                    "can_run_low": gpu["free_gb"] >= 10,
                }
    except Exception as e:
        logger.error(f"GPU Detection error: {e}")

    return {
        "available": False,
        "gpu": None,
        "recommended_mode": "low",
        "can_run_full": False,
        "can_run_low": False,
    }


# ============================================================================
# Audio Duration Helper (Kept for compatibility if needed, though less relevant for LLM)
# ============================================================================


def get_audio_duration(audio_path: Path) -> Optional[float]:
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
        logger.info(f"Detected: {gpu_info['gpu']['name']}")
        logger.info(
            f"VRAM: {gpu_info['gpu']['free_gb']}GB free / {gpu_info['gpu']['total_gb']}GB total"
        )
        logger.info(f"Recommended mode: {gpu_info['recommended_mode']}")
    else:
        logger.warning("No NVIDIA GPU detected or nvidia-smi not available")


def refresh_gpu_info() -> Dict[str, Any]:
    """Refresh GPU info and return updated data"""
    global gpu_info
    gpu_info = get_gpu_info()
    return gpu_info
