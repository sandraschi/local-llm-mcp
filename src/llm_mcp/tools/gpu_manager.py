"""GPU Manager tool for Local LLM MCP server.

Specialized tool for monitoring and controlling NVIDIA GPU resources,
particularly optimized for RTX 4090 to prevent memory issues and "gumming up".
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from llm_mcp.utils.logging import get_logger

logger = get_logger(__name__)

# Import FastMCP components
try:
    from fastmcp import FastMCP
    from fastmcp.tools import Tool
    FASTMCP_AVAILABLE = True
except ImportError:
    logger.error("FastMCP not available - GPU manager requires FastMCP >= 2.12.0")
    FASTMCP_AVAILABLE = False

# GPU monitoring dependencies
try:
    import GPUtil
    import torch
    import psutil
    GPU_DEPS_AVAILABLE = True
except ImportError:
    GPU_DEPS_AVAILABLE = False
    logger.warning("GPU monitoring dependencies not available. Install with: pip install gputil torch psutil")

@dataclass
class GPUStatus:
    """GPU status information."""
    id: int
    name: str
    memory_used: float
    memory_total: float
    memory_free: float
    memory_utilization: float
    gpu_utilization: float
    temperature: float
    fan_speed: Optional[float] = None
    power_usage: Optional[float] = None
    power_limit: Optional[float] = None

async def get_gpu_status() -> List[GPUStatus]:
    """Get comprehensive GPU status information."""
    if not GPU_DEPS_AVAILABLE:
        return []

    try:
        gpus = GPUtil.getGPUs()
        gpu_statuses = []

        for i, gpu in enumerate(gpus):
            # Get additional info from torch if available
            memory_info = None
            if torch.cuda.is_available() and i < torch.cuda.device_count():
                try:
                    torch.cuda.synchronize(i)  # Sync before checking memory
                    memory_info = torch.cuda.mem_get_info(i)
                    memory_free, memory_total = memory_info
                    memory_used = memory_total - memory_free
                    memory_utilization = (memory_used / memory_total) * 100
                except Exception as e:
                    logger.warning(f"Failed to get CUDA memory info for GPU {i}: {e}")
                    memory_used = gpu.memoryUsed
                    memory_total = gpu.memoryTotal
                    memory_free = gpu.memoryFree
                    memory_utilization = gpu.memoryUtil * 100
            else:
                memory_used = gpu.memoryUsed
                memory_total = gpu.memoryTotal
                memory_free = gpu.memoryFree
                memory_utilization = gpu.memoryUtil * 100

            gpu_status = GPUStatus(
                id=i,
                name=gpu.name,
                memory_used=memory_used,
                memory_total=memory_total,
                memory_free=memory_free,
                memory_utilization=memory_utilization,
                gpu_utilization=gpu.load * 100,
                temperature=gpu.temperature,
                fan_speed=getattr(gpu, 'fan_speed', None),
                power_usage=getattr(gpu, 'power_usage', None),
                power_limit=getattr(gpu, 'power_limit', None),
            )
            gpu_statuses.append(gpu_status)

        return gpu_statuses

    except Exception as e:
        logger.error(f"Failed to get GPU status: {e}")
        return []

async def clear_gpu_memory(gpu_id: int = 0, force: bool = False) -> Dict[str, Any]:
    """Clear GPU memory by running garbage collection and cache clearing."""
    if not GPU_DEPS_AVAILABLE:
        return {"error": "GPU dependencies not available"}

    try:
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        if gpu_id >= torch.cuda.device_count():
            return {"error": f"GPU {gpu_id} not available"}

        # Switch to the target GPU
        torch.cuda.set_device(gpu_id)

        # Get memory before cleanup
        memory_before = torch.cuda.mem_get_info(gpu_id)

        # Force garbage collection
        import gc
        gc.collect()

        # Clear CUDA cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize(gpu_id)

        # Get memory after cleanup
        memory_after = torch.cuda.mem_get_info(gpu_id)

        memory_freed = memory_before[0] - memory_after[0]  # free_memory difference

        return {
            "success": True,
            "gpu_id": gpu_id,
            "memory_before": {
                "free": memory_before[0],
                "total": memory_before[1]
            },
            "memory_after": {
                "free": memory_after[0],
                "total": memory_after[1]
            },
            "memory_freed": memory_freed,
            "memory_freed_mb": memory_freed / (1024 * 1024),
            "summary": f"Freed {memory_freed / (1024 * 1024):.1f}MB on GPU {gpu_id}"
        }

    except Exception as e:
        logger.error(f"Failed to clear GPU memory: {e}")
        return {"error": f"Failed to clear GPU memory: {str(e)}"}

async def optimize_gpu_memory(gpu_id: int = 0) -> Dict[str, Any]:
    """Optimize GPU memory usage with advanced techniques."""
    if not GPU_DEPS_AVAILABLE:
        return {"error": "GPU dependencies not available"}

    try:
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        if gpu_id >= torch.cuda.device_count():
            return {"error": f"GPU {gpu_id} not available"}

        torch.cuda.set_device(gpu_id)

        # Get initial memory
        initial_memory = torch.cuda.mem_get_info(gpu_id)

        # Multiple cleanup passes
        import gc
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(gpu_id)
            await asyncio.sleep(0.1)  # Small delay between passes

        # Final memory check
        final_memory = torch.cuda.mem_get_info(gpu_id)
        memory_freed = initial_memory[0] - final_memory[0]

        # Get GPU status
        gpu_statuses = await get_gpu_status()
        current_status = next((s for s in gpu_statuses if s.id == gpu_id), None)

        return {
            "success": True,
            "gpu_id": gpu_id,
            "optimization_performed": True,
            "memory_freed_mb": memory_freed / (1024 * 1024),
            "current_memory_utilization": current_status.memory_utilization if current_status else None,
            "current_gpu_utilization": current_status.gpu_utilization if current_status else None,
            "temperature": current_status.temperature if current_status else None,
            "recommendations": _generate_gpu_recommendations(current_status) if current_status else []
        }

    except Exception as e:
        logger.error(f"Failed to optimize GPU memory: {e}")
        return {"error": f"Failed to optimize GPU memory: {str(e)}"}

def _generate_gpu_recommendations(status: GPUStatus) -> List[str]:
    """Generate GPU optimization recommendations based on current status."""
    recommendations = []

    if status.memory_utilization > 95:
        recommendations.append("CRITICAL: GPU memory nearly full. Consider unloading models or reducing batch sizes.")
    elif status.memory_utilization > 85:
        recommendations.append("WARNING: High GPU memory usage. Monitor closely.")

    if status.temperature > 85:
        recommendations.append("WARNING: High GPU temperature. Ensure proper cooling.")
    elif status.temperature > 95:
        recommendations.append("CRITICAL: GPU overheating. Immediate cooling action required.")

    if status.gpu_utilization < 10 and status.memory_utilization > 50:
        recommendations.append("Low GPU utilization but high memory usage. Consider memory optimization.")

    if not recommendations:
        recommendations.append("GPU status looks good. No immediate action required.")

    return recommendations

async def monitor_gpu_health(gpu_id: int = 0) -> Dict[str, Any]:
    """Monitor GPU health and provide detailed diagnostics."""
    if not GPU_DEPS_AVAILABLE:
        return {"error": "GPU dependencies not available"}

    try:
        gpu_statuses = await get_gpu_status()

        if gpu_id >= len(gpu_statuses):
            return {"error": f"GPU {gpu_id} not available"}

        status = gpu_statuses[gpu_id]

        # Additional system info
        system_memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)

        health_score = 100

        # Memory health (40% weight)
        if status.memory_utilization > 95:
            health_score -= 40
        elif status.memory_utilization > 85:
            health_score -= 20
        elif status.memory_utilization > 70:
            health_score -= 10

        # Temperature health (30% weight)
        if status.temperature > 90:
            health_score -= 30
        elif status.temperature > 80:
            health_score -= 15
        elif status.temperature > 70:
            health_score -= 5

        # Utilization balance (20% weight)
        if status.gpu_utilization > 95:
            health_score -= 10  # Overutilized
        elif status.gpu_utilization < 5 and status.memory_utilization > 50:
            health_score -= 10  # Underutilized with high memory

        # System resources (10% weight)
        if system_memory.percent > 90:
            health_score -= 10

        return {
            "success": True,
            "gpu_id": gpu_id,
            "health_score": max(0, health_score),
            "health_status": "critical" if health_score < 30 else "warning" if health_score < 70 else "good",
            "gpu_status": {
                "memory_utilization": status.memory_utilization,
                "gpu_utilization": status.gpu_utilization,
                "temperature": status.temperature,
                "memory_used_gb": status.memory_used / (1024**3),
                "memory_total_gb": status.memory_total / (1024**3)
            },
            "system_status": {
                "cpu_percent": cpu_percent,
                "system_memory_percent": system_memory.percent,
                "system_memory_used_gb": system_memory.used / (1024**3),
                "system_memory_total_gb": system_memory.total / (1024**3)
            },
            "recommendations": _generate_gpu_recommendations(status),
            "alerts": [
                f"High memory usage ({status.memory_utilization:.1f}%)" if status.memory_utilization > 85 else None,
                f"High temperature ({status.temperature}°C)" if status.temperature > 85 else None,
                f"High system memory ({system_memory.percent}%)" if system_memory.percent > 90 else None
            ],
            "alerts": [alert for alert in [
                f"High memory usage ({status.memory_utilization:.1f}%)" if status.memory_utilization > 85 else None,
                f"High temperature ({status.temperature}°C)" if status.temperature > 85 else None,
                f"High system memory ({system_memory.percent}%)" if system_memory.percent > 90 else None
            ] if alert is not None]
        }

    except Exception as e:
        logger.error(f"Failed to monitor GPU health: {e}")
        return {"error": f"Failed to monitor GPU health: {str(e)}"}

# Register tools with MCP
def register_gpu_manager_tools(mcp, register_individual_tools: bool = True):
    """Register GPU Manager tools with the MCP server.

    Args:
        mcp: The MCP server instance
        register_individual_tools: Whether to register individual GPU tools (default: True)
    """
    if not FASTMCP_AVAILABLE:
        logger.error("Cannot register GPU manager tools - FastMCP not available")
        return mcp

    # Register gpu_status conditionally
    if register_individual_tools:
        @mcp.tool()
        async def gpu_status() -> Dict[str, Any]:
        """Get comprehensive status of all NVIDIA GPUs.

        Returns detailed information about GPU memory, utilization, temperature,
        and other vital statistics for monitoring and optimization.
        """
        try:
            gpu_statuses = await get_gpu_status()

            if not gpu_statuses:
                return {
                    "success": False,
                    "error": "No GPUs detected or GPU monitoring not available",
                    "troubleshooting": [
                        "Ensure NVIDIA drivers are installed",
                        "Install GPU monitoring dependencies: pip install gputil torch",
                        "Check GPU is properly connected and powered"
                    ]
                }

            return {
                "success": True,
                "gpu_count": len(gpu_statuses),
                "gpus": [
                    {
                        "id": gpu.id,
                        "name": gpu.name,
                        "memory": {
                            "used_gb": round(gpu.memory_used / (1024**3), 2),
                            "total_gb": round(gpu.memory_total / (1024**3), 2),
                            "free_gb": round(gpu.memory_free / (1024**3), 2),
                            "utilization_percent": round(gpu.memory_utilization, 1)
                        },
                        "utilization_percent": round(gpu.gpu_utilization, 1),
                        "temperature_celsius": gpu.temperature,
                        "fan_speed_percent": gpu.fan_speed,
                        "power": {
                            "usage_watts": gpu.power_usage,
                            "limit_watts": gpu.power_limit
                        } if gpu.power_usage and gpu.power_limit else None
                    }
                    for gpu in gpu_statuses
                ],
                "summary": {
                    "total_memory_gb": sum(gpu.memory_total / (1024**3) for gpu in gpu_statuses),
                    "used_memory_gb": sum(gpu.memory_used / (1024**3) for gpu in gpu_statuses),
                    "average_utilization": sum(gpu.gpu_utilization for gpu in gpu_statuses) / len(gpu_statuses),
                    "hottest_gpu_temp": max(gpu.temperature for gpu in gpu_statuses)
                }
            }

        except Exception as e:
            logger.error(f"Failed to get GPU status: {e}")
            return {"error": f"Failed to get GPU status: {str(e)}"}

    # Register individual GPU tools conditionally
    if register_individual_tools:
        @mcp.tool()
        async def gpu_clear_memory(gpu_id: int = 0) -> Dict[str, Any]:
            """Clear GPU memory to prevent memory fragmentation and "gumming up".

            This tool performs comprehensive memory cleanup including:
            - Python garbage collection
            - CUDA cache clearing
            - Memory defragmentation

            Especially useful for RTX 4090 GPUs that can accumulate memory fragmentation
            during intensive AI workloads.
            """
            return await clear_gpu_memory(gpu_id)

        @mcp.tool()
        async def gpu_optimize(gpu_id: int = 0) -> Dict[str, Any]:
            """Perform advanced GPU memory optimization.

            Runs multiple optimization passes and provides detailed before/after metrics
            to ensure the GPU is running optimally, especially important for high-end
            GPUs like the RTX 4090.
            """
            return await optimize_gpu_memory(gpu_id)

        @mcp.tool()
        async def gpu_health_check(gpu_id: int = 0) -> Dict[str, Any]:
            """Comprehensive GPU health monitoring and diagnostics.

            Provides detailed health assessment with recommendations for maintaining
            optimal GPU performance, particularly crucial for preventing issues with
            high-performance GPUs like the RTX 4090.
            """
            return await monitor_gpu_health(gpu_id)

    logger.info("Registered GPU Manager tools")
    return mcp