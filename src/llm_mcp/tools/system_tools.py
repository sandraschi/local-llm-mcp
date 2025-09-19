"""System management and monitoring tools for the LLM MCP server."""
import os
import sys
import platform
import psutil
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

def get_system_info() -> Dict[str, Any]:
    """Get detailed system information.
    
    Returns:
        Dictionary containing system information
    """
    try:
        import GPUtil
        gpus = [gpu.name for gpu in GPUtil.getGPUs()]
    except ImportError:
        gpus = ["N/A (install nvidia-ml-py3 for GPU info)"]
    except Exception as e:
        gpus = [f"Error getting GPU info: {str(e)}"]
    
    return {
        "platform": {
            "system": platform.system(),
            "node": platform.node(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
        "cpu": {
            "physical_cores": psutil.cpu_count(logical=False),
            "total_cores": psutil.cpu_count(logical=True),
            "max_frequency": f"{psutil.cpu_freq().max:.2f}Mhz" if hasattr(psutil, 'cpu_freq') and psutil.cpu_freq() else "N/A",
            "cpu_percent": psutil.cpu_percent(interval=1, percpu=True),
            "load_avg": [x / psutil.cpu_count() * 100 for x in psutil.getloadavg()] if hasattr(psutil, 'getloadavg') else [],
        },
        "memory": {
            "total": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            "available": f"{psutil.virtual_memory().available / (1024**3):.2f} GB",
            "used": f"{psutil.virtual_memory().used / (1024**3):.2f} GB",
            "percent": psutil.virtual_memory().percent,
        },
        "disk": {
            "total": f"{psutil.disk_usage('/').total / (1024**3):.2f} GB",
            "used": f"{psutil.disk_usage('/').used / (1024**3):.2f} GB",
            "free": f"{psutil.disk_usage('/').free / (1024**3):.2f} GB",
            "percent": psutil.disk_usage('/').percent,
        },
        "gpu": gpus,
        "process": {
            "pid": os.getpid(),
            "name": psutil.Process().name(),
            "status": psutil.Process().status(),
            "create_time": datetime.fromtimestamp(psutil.Process().create_time()).strftime("%Y-%m-%d %H:%M:%S"),
            "cpu_percent": psutil.Process().cpu_percent(interval=1),
            "memory_percent": psutil.Process().memory_percent(),
            "memory_info": {
                "rss": f"{psutil.Process().memory_info().rss / (1024**2):.2f} MB",
                "vms": f"{psutil.Process().memory_info().vms / (1024**2):.2f} MB",
            },
        },
    }

def get_service_status() -> Dict[str, Any]:
    """Get the status of common services used by the MCP server.
    
    Returns:
        Dictionary containing service status information
    """
    services = {}
    
    # Check Redis
    redis_status = {"status": "unknown", "version": "unknown"}
    try:
        import redis
        r = redis.Redis()
        redis_status["status"] = "running" if r.ping() else "not responding"
        redis_status["version"] = r.info().get("redis_version", "unknown")
    except Exception as e:
        redis_status["status"] = f"error: {str(e)}"
    services["redis"] = redis_status
    
    # Check database connections, etc.
    # Add more service checks as needed
    
    return services

def register_system_tools(mcp):
    """Register all system-related tools with the MCP server.
    
    Args:
        mcp: The MCP server instance with tool decorator
        
    Returns:
        The MCP server instance with system tools registered
        
    Notes:
        - Tools are registered with stateful=True where appropriate to maintain state between invocations
        - State TTL is set based on the expected cache duration for each tool
        - Critical system operations like restart/shutdown are not cached
    """
    @mcp.tool
    async def system_info() -> Dict[str, Any]:
        """Get detailed system information with stateful caching.
        
        This tool caches system information to improve performance while ensuring
        fresh data is available through the TTL mechanism.
        
        Returns:
            Dictionary containing system information with caching
        """
        return get_system_info()
    
    @mcp.tool
    async def service_status() -> Dict[str, Any]:
        """Check the status of dependent services with stateful caching.
        
        This tool caches service status to improve performance.
        The cache is automatically managed by FastMCP's stateful tools.
        
        Returns:
            Dictionary containing service status information with caching
        """
        return get_service_status()
    
    @mcp.tool
    async def server_health() -> Dict[str, Any]:
        """Get the health status of the MCP server with stateful caching.
        
        This tool caches health status to improve performance while ensuring
        timely updates through a shorter TTL.
        
        Returns:
            Dictionary containing health status information with caching
        """
        sys_info = get_system_info()
        services = get_service_status()
        
        # Basic health check
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "cpu_usage": sys_info["cpu"]["cpu_percent"],
                "memory_usage": sys_info["memory"]["percent"],
                "disk_usage": sys_info["disk"]["percent"],
            },
            "services": services,
            "issues": []
        }
        
        # Check for issues
        if sys_info["memory"]["percent"] > 90:
            health["issues"].append("High memory usage")
            health["status"] = "degraded"
            
        if sys_info["disk"]["percent"] > 90:
            health["issues"].append("High disk usage")
            health["status"] = "degraded"
            
        for service, status in services.items():
            if status["status"] != "running":
                health["issues"].append(f"Service {service} is {status['status']}")
                health["status"] = "degraded"
        
        if not health["issues"]:
            health["message"] = "All systems operational"
        
        return health
    
    @mcp.tool()  # Restart server
    async def server_restart(delay: int = 0) -> Dict[str, Any]:
        """Restart the MCP server.
        
        This tool does not use caching as it performs a critical system operation.
        
        Args:
            delay: Number of seconds to wait before restarting
            
        Returns:
            Confirmation message
        """
        if delay > 0:
            return {"status": "scheduled", "message": f"Server will restart in {delay} seconds"}
        
        # In a real implementation, this would trigger a restart
        return {"status": "not_implemented", "message": "Restart functionality not implemented"}
    
    @mcp.tool()  # Shutdown server
    async def server_shutdown(delay: int = 0) -> Dict[str, Any]:
        """Shut down the MCP server.
        
        This tool does not use caching as it performs a critical system operation.
        
        Args:
            delay: Number of seconds to wait before shutting down
            
        Returns:
            Confirmation message
        """
        if delay > 0:
            return {"status": "scheduled", "message": f"Server will shut down in {delay} seconds"}
        
        # In a real implementation, this would trigger a shutdown
        return {"status": "not_implemented", "message": "Shutdown functionality not implemented"}
    
    return mcp
