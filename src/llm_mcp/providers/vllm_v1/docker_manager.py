"""Docker container management for vLLM v0.10.1.1."""

import asyncio
import aiohttp
import docker
import logging
import time
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class VLLMDockerManager:
    """Manage vLLM v0.10.1.1 Docker containers with RTX 4090 optimization."""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.container_name = "local-llm-mcp-vllm-v10"
        self.container = None
        self.base_url = "http://localhost:8000"
        
    async def start_vllm_container(
        self,
        model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
        tensor_parallel_size: int = 1,
        port: int = 8000,
        **kwargs
    ) -> Dict[str, Any]:
        """Start vLLM v0.10.1.1 Docker container with RTX 4090 optimization."""
        
        # Check if container is already running
        existing = self._get_existing_container()
        if existing and existing.status == "running":
            logger.info("vLLM container already running")
            return {"status": "already_running", "container_id": existing.id}
        
        # Clean up any existing stopped container
        if existing:
            existing.remove()
            
        # RTX 4090 optimized environment variables
        environment = {
            # V1 Engine Configuration
            "VLLM_USE_V1": "1",
            "VLLM_ATTENTION_BACKEND": "FLASHINFER",
            "VLLM_ENABLE_PREFIX_CACHING": "1",
            "VLLM_GPU_MEMORY_UTILIZATION": str(gpu_memory_utilization),
            
            # RTX 4090 Specific (Ada Lovelace)
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:128",
            "TORCH_CUDA_ARCH_LIST": "8.9",
            
            # FlashAttention Configuration
            "VLLM_FLASH_ATTN_VERSION": "2",
            
            # Performance Settings
            "VLLM_MAX_NUM_BATCHED_TOKENS": "8192",
            "VLLM_MAX_NUM_SEQS": "256",
        }
        
        # Container configuration
        container_config = {
            "image": "local-llm-mcp/vllm-v10:latest",
            "name": self.container_name,
            "environment": environment,
            "ports": {8000: port},
            "detach": True,
            "device_requests": [
                docker.types.DeviceRequest(count=1, capabilities=[["gpu"]])
            ],
            "ipc_mode": "host",  # Required for shared memory optimization
            "volumes": {
                str(Path.home() / ".cache" / "huggingface"): {
                    "bind": "/root/.cache/huggingface", 
                    "mode": "rw"
                }
            }
        }
        
        # vLLM server command with V1 engine optimizations
        command = [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--model", model_id,
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--max-model-len", str(max_model_len),
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--enable-prefix-caching",
            "--max-num-batched-tokens", "8192",
            "--trust-remote-code",
            "--served-model-name", model_id.split("/")[-1] if "/" in model_id else model_id
        ]
        
        logger.info(f"Starting vLLM v0.10.1.1 container with model: {model_id}")
        logger.info(f"Command: {' '.join(command)}")
        
        try:
            # Start the container
            self.container = self.docker_client.containers.run(
                command=command,
                **container_config
            )
            
            logger.info(f"vLLM container started: {self.container.id[:12]}")
            
            # Wait for the server to be ready
            await self._wait_for_health_check(timeout=300)  # 5 minutes for model loading
            
            return {
                "status": "started",
                "container_id": self.container.id,
                "model": model_id,
                "base_url": f"http://localhost:{port}",
                "gpu_memory_utilization": gpu_memory_utilization,
                "tensor_parallel_size": tensor_parallel_size
            }
            
        except Exception as e:
            logger.error(f"Failed to start vLLM container: {e}")
            if self.container:
                try:
                    self.container.remove(force=True)
                except:
                    pass
            raise
    
    async def stop_vllm_container(self) -> Dict[str, Any]:
        """Stop the vLLM Docker container."""
        container = self._get_existing_container()
        if not container:
            return {"status": "not_running"}
        
        try:
            container.stop(timeout=30)
            container.remove()
            self.container = None
            logger.info("vLLM container stopped and removed")
            return {"status": "stopped"}
        except Exception as e:
            logger.error(f"Failed to stop vLLM container: {e}")
            return {"status": "error", "error": str(e)}
    
    async def get_container_status(self) -> Dict[str, Any]:
        """Get the status of the vLLM container."""
        container = self._get_existing_container()
        if not container:
            return {"status": "not_running"}
        
        try:
            container.reload()
            stats = container.stats(stream=False)
            
            # Calculate GPU memory usage if available
            gpu_usage = None
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_usage = {
                    "used_gb": mem_info.used / (1024**3),
                    "total_gb": mem_info.total / (1024**3),
                    "utilization_percent": (mem_info.used / mem_info.total) * 100
                }
            except:
                pass
            
            return {
                "status": container.status,
                "container_id": container.id[:12],
                "image": container.image.tags[0] if container.image.tags else "unknown",
                "ports": container.ports,
                "gpu_usage": gpu_usage,
                "cpu_usage_percent": self._calculate_cpu_usage(stats),
                "memory_usage_mb": stats["memory_stats"]["usage"] / (1024**2) if "memory_stats" in stats else None,
            }
            
        except Exception as e:
            logger.error(f"Failed to get container status: {e}")
            return {"status": "error", "error": str(e)}
    
    async def get_container_logs(self, tail: int = 100) -> str:
        """Get recent logs from the vLLM container."""
        container = self._get_existing_container()
        if not container:
            return "Container not found"
        
        try:
            logs = container.logs(tail=tail, timestamps=True).decode('utf-8')
            return logs
        except Exception as e:
            return f"Error getting logs: {e}"
    
    async def test_vllm_api(self) -> Dict[str, Any]:
        """Test the vLLM API endpoints."""
        try:
            async with aiohttp.ClientSession() as session:
                # Test health endpoint
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status != 200:
                        return {"status": "unhealthy", "health_status": response.status}
                
                # Test models endpoint  
                async with session.get(f"{self.base_url}/v1/models") as response:
                    if response.status == 200:
                        models = await response.json()
                        return {
                            "status": "healthy",
                            "available_models": [m["id"] for m in models.get("data", [])],
                            "api_version": "v1"
                        }
                    else:
                        return {"status": "api_error", "models_status": response.status}
                        
        except Exception as e:
            return {"status": "connection_error", "error": str(e)}
    
    async def benchmark_performance(
        self, 
        test_prompt: str = "Hello, how are you today?",
        max_tokens: int = 50
    ) -> Dict[str, Any]:
        """Benchmark vLLM v0.10.1.1 performance."""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                    "messages": [{"role": "user", "content": test_prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                }
                
                start_time = time.time()
                async with session.post(f"{self.base_url}/v1/chat/completions", json=payload) as response:
                    end_time = time.time()
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # Calculate performance metrics
                        duration = end_time - start_time
                        completion = result["choices"][0]["message"]["content"]
                        tokens_generated = len(completion.split())  # Rough estimate
                        
                        return {
                            "status": "success",
                            "duration_seconds": duration,
                            "tokens_generated": tokens_generated,
                            "tokens_per_second": tokens_generated / duration if duration > 0 else 0,
                            "prompt_tokens": result["usage"]["prompt_tokens"],
                            "completion_tokens": result["usage"]["completion_tokens"],
                            "total_tokens": result["usage"]["total_tokens"],
                            "response_preview": completion[:100] + "..." if len(completion) > 100 else completion
                        }
                    else:
                        return {"status": "api_error", "http_status": response.status}
                        
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _get_existing_container(self):
        """Get existing container by name."""
        try:
            return self.docker_client.containers.get(self.container_name)
        except docker.errors.NotFound:
            return None
    
    async def _wait_for_health_check(self, timeout: int = 300):
        """Wait for vLLM server to be healthy."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result = await self.test_vllm_api()
                if result.get("status") == "healthy":
                    logger.info("vLLM v0.10.1.1 server is healthy and ready")
                    return
                    
                logger.info(f"Waiting for vLLM server... Status: {result.get('status')}")
                
            except Exception as e:
                logger.debug(f"Health check failed: {e}")
            
            await asyncio.sleep(10)
        
        raise TimeoutError(f"vLLM server did not become healthy within {timeout} seconds")
    
    def _calculate_cpu_usage(self, stats: Dict) -> Optional[float]:
        """Calculate CPU usage percentage from Docker stats."""
        try:
            cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - stats["precpu_stats"]["cpu_usage"]["total_usage"]
            system_delta = stats["cpu_stats"]["system_cpu_usage"] - stats["precpu_stats"]["system_cpu_usage"]
            num_cpus = stats["cpu_stats"]["online_cpus"]
            
            if system_delta > 0:
                return (cpu_delta / system_delta) * num_cpus * 100.0
        except (KeyError, ZeroDivisionError):
            pass
        return None
