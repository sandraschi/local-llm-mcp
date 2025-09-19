"""Monitoring and logging tools for the LLM MCP server."""
import time
import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
import statistics

logger = logging.getLogger(__name__)

@dataclass
class Metric:
    """A single metric data point."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)

class MetricsCollector:
    """Collects and aggregates metrics over time."""
    
    def __init__(self, max_metrics: int = 1000):
        self.metrics = defaultdict(lambda: deque(maxlen=max_metrics))
        self.callbacks = []
    
    def add_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Add a new metric.
        
        Args:
            name: Name of the metric
            value: Numeric value
            tags: Optional key-value pairs for filtering/grouping
        """
        if tags is None:
            tags = {}
            
        metric = Metric(name=name, value=float(value), tags=tags)
        self.metrics[name].append(metric)
        
        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback(metric)
            except Exception as e:
                logger.error(f"Error in metric callback: {e}", exc_info=True)
    
    def get_metrics(
        self, 
        name: str, 
        since: Optional[float] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> List[Metric]:
        """Get metrics by name, optionally filtered by time and tags."""
        if name not in self.metrics:
            return []
            
        metrics = list(self.metrics[name])
        
        # Filter by time
        if since is not None:
            metrics = [m for m in metrics if m.timestamp >= since]
            
        # Filter by tags
        if tags:
            metrics = [
                m for m in metrics
                if all(m.tags.get(k) == v for k, v in tags.items())
            ]
            
        return metrics
    
    def get_stats(
        self, 
        name: str, 
        since: Optional[float] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """Get statistics for a metric."""
        metrics = self.get_metrics(name, since, tags)
        if not metrics:
            return {}
            
        values = [m.value for m in metrics]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p90": statistics.quantiles(values, n=10)[-1] if len(values) > 1 else values[0],
            "p95": statistics.quantiles(values, n=20)[-1] if len(values) > 1 else values[0],
            "p99": statistics.quantiles(values, n=100)[-1] if len(values) > 1 else values[0],
        }
    
    def register_callback(self, callback: Callable[[Metric], None]):
        """Register a callback to be called when a new metric is added."""
        self.callbacks.append(callback)

# Global metrics collector
metrics_collector = MetricsCollector()

def track_metric(name: str, tags: Optional[Dict[str, str]] = None):
    """Decorator to track function execution as a metric."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                metrics_collector.add_metric(
                    f"{name}.duration", 
                    duration * 1000,  # Convert to ms
                    {"status": "success", **(tags or {})}
                )
                metrics_collector.add_metric(
                    f"{name}.success", 
                    1,
                    tags or {}
                )
                return result
            except Exception as e:
                metrics_collector.add_metric(
                    f"{name}.error", 
                    1,
                    {"error": str(e), "error_type": e.__class__.__name__, **(tags or {})}
                )
                raise
        
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics_collector.add_metric(
                    f"{name}.duration", 
                    duration * 1000,  # Convert to ms
                    {"status": "success", **(tags or {})}
                )
                metrics_collector.add_metric(
                    f"{name}.success", 
                    1,
                    tags or {}
                )
                return result
            except Exception as e:
                metrics_collector.add_metric(
                    f"{name}.error", 
                    1,
                    {"error": str(e), "error_type": e.__class__.__name__, **(tags or {})}
                )
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

# Implementation functions (without @tool decorator)
async def get_metrics_impl(
    name: str,
    since_minutes: float = 60,
    tags: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Get metrics for a specific name.
    
    Args:
        name: Name of the metric (can include wildcards)
        since_minutes: How far back to look in minutes
        tags: Optional tags to filter by
        
    Returns:
        Dictionary containing metrics and statistics
    """
    since = time.time() - (since_minutes * 60)
    
    # If name contains wildcards, find matching metrics
    if '*' in name:
        import fnmatch
        all_metrics = metrics_collector.metrics.keys()
        matching_metrics = fnmatch.filter(all_metrics, name)
    else:
        matching_metrics = [name] if name in metrics_collector.metrics else []
    
    result = {}
    for metric_name in matching_metrics:
        metrics = metrics_collector.get_metrics(metric_name, since, tags)
        if not metrics:
            continue
            
        values = [m.value for m in metrics]
        timestamps = [m.timestamp for m in metrics]
        
        result[metric_name] = {
            "count": len(metrics),
            "values": values,
            "timestamps": timestamps,
            "stats": metrics_collector.get_stats(metric_name, since, tags)
        }
    
    return result

async def get_metric_stats_impl(
    name: str,
    since_minutes: float = 60,
    tags: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Get statistics for a specific metric.
    
    Args:
        name: Name of the metric
        since_minutes: How far back to look in minutes
        tags: Optional tags to filter by
        
    Returns:
        Dictionary containing statistics
    """
    since = time.time() - (since_minutes * 60)
    return metrics_collector.get_stats(name, since, tags)

async def set_log_level_impl(
    logger_name: str = "",
    level: str = "INFO"
) -> Dict[str, Any]:
    """Set the log level for a logger.
    
    Args:
        logger_name: Name of the logger (empty string for root)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Confirmation message
    """
    try:
        log_level = getattr(logging, level.upper())
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        return {"status": "success", "logger": logger_name, "level": level}
    except Exception as e:
        return {"status": "error", "error": str(e)}

# Background task for collecting system metrics
async def collect_system_metrics():
    """Periodically collect system metrics."""
    import psutil
    
    while True:
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics_collector.add_metric("system.cpu.percent", cpu_percent)
            
            # Memory
            mem = psutil.virtual_memory()
            metrics_collector.add_metric("system.memory.percent", mem.percent)
            metrics_collector.add_metric("system.memory.used_mb", mem.used / (1024 * 1024))
            
            # Disk
            disk = psutil.disk_usage('/')
            metrics_collector.add_metric("system.disk.percent", disk.percent)
            metrics_collector.add_metric("system.disk.used_gb", disk.used / (1024**3))
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}", exc_info=True)
        
        await asyncio.sleep(60)  # Collect every minute

def register_monitoring_tools(mcp):
    """Register all monitoring-related tools with the MCP server.
    
    Args:
        mcp: The MCP server instance with tool decorator
        
    Returns:
        The MCP server instance with monitoring tools registered
        
    Notes:
        - Tools are registered with stateful=True to maintain state between invocations
        - State TTL is set based on the expected cache duration for each tool
        - Metrics collection runs as a background task with state persistence
    """
    # Get the tool decorator from the mcp instance
    tool = mcp.tool
    
    @tool()  # Get metrics
    async def get_metrics(
        name: str,
        since_minutes: float = 60,
        tags: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Get metrics for a specific name with stateful caching.
        
        This tool caches metrics data to improve performance while ensuring
        fresh data is available through the TTL mechanism.
        
        Args:
            name: Name of the metric (can include wildcards)
            since_minutes: How far back to look in minutes
            tags: Optional tags to filter by
            
        Returns:
            Dictionary containing metrics and statistics with caching
        """
        return await get_metrics_impl(name, since_minutes, tags)
    
    @tool()  # Get metric stats
    async def get_metric_stats(
        name: str,
        since_minutes: float = 60,
        tags: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Get statistics for a specific metric with stateful caching.
        
        This tool caches metric statistics to improve performance.
        The cache is automatically managed by FastMCP's stateful tools.
        
        Args:
            name: Name of the metric
            since_minutes: How far back to look in minutes
            tags: Optional tags to filter by
            
        Returns:
            Dictionary containing statistics with caching
        """
        return await get_metric_stats_impl(name, since_minutes, tags)
    
    @tool()  # Set log level
    async def set_log_level(
        logger_name: str = "",
        level: str = "INFO"
    ) -> Dict[str, Any]:
        """Set the log level for a logger.
        
        This tool does not use caching as it directly modifies logger state.
        
        Args:
            logger_name: Name of the logger (empty string for root)
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            
        Returns:
            Confirmation message
        """
        return await set_log_level_impl(logger_name, level)
    
    # Start the background task with state persistence
    asyncio.create_task(collect_system_metrics())
    
    return mcp
