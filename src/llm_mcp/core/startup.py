"""Application startup and shutdown handlers."""
from typing import Any, Dict

from fastapi import FastAPI
from fastmcp import FastMCP

from .config import Settings, get_settings
from ..services.model_manager import ModelManager
from ..models.base import ModelStatus, ModelProvider

async def startup_event(app: FastAPI) -> None:
    """Initialize application services on startup."""
    settings = get_settings()
    
    # Initialize model manager
    model_manager = ModelManager(settings.providers.dict())
    
    # Store in app state
    app.state.model_manager = model_manager
    
    # Initialize model manager
    await model_manager.initialize()
    
    print("Application startup complete")

async def shutdown_event(app: FastAPI) -> None:
    """Clean up resources on shutdown."""
    # Clean up model manager
    if hasattr(app.state, 'model_manager'):
        await app.state.model_manager.close()
    
    print("Application shutdown complete")

def setup_mcp(mcp: FastMCP) -> None:
    """Set up FastMCP-specific configurations and register tools."""
    settings = get_settings()
    
    @mcp.tool()
    async def list_models() -> list[dict[str, Any]]:
        """List all available models from all providers."""
        model_manager = mcp.app.state.model_manager
        models = await model_manager.list_models()
        return [{"id": m.id, "name": m.name, "provider": m.provider.value} for m in models]
    
    @mcp.tool()
    async def get_model(model_id: str) -> dict[str, Any]:
        """Get details about a specific model.
        
        Args:
            model_id: The ID of the model to retrieve
            
        Returns:
            Model details including name, provider, status, and capabilities
        """
        model_manager = mcp.app.state.model_manager
        model = await model_manager.get_model(model_id)
        if model:
            return {
                "id": model.id,
                "name": model.name,
                "provider": model.provider.value,
                "status": model.status.value,
                "capabilities": [c.value for c in model.capabilities]
            }
        return {}
    
    @mcp.tool()
    async def generate_text(
        model_id: str,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate text using the specified model.
        
        Args:
            model_id: The ID of the model to use
            prompt: The input prompt for text generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Nucleus sampling parameter
            frequency_penalty: Penalty for frequent tokens
            presence_penalty: Penalty for new tokens
            stop: List of strings that stop generation when encountered
            
        Returns:
            Generated text and metadata
        """
        model_manager = mcp.app.state.model_manager
        result = await model_manager.generate_text(
            model_id=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
        )
        return {"text": result.text, "metadata": result.metadata}
    
    @mcp.tool()
    async def chat(
        model_id: str,
        messages: list[dict[str, str]],
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate a chat completion using the specified model.
        
        Args:
            model_id: The ID of the model to use
            messages: List of message objects with 'role' and 'content'
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Nucleus sampling parameter
            frequency_penalty: Penalty for frequent tokens
            presence_penalty: Penalty for new tokens
            stop: List of strings that stop generation when encountered
            
        Returns:
            Generated response and metadata
        """
        model_manager = mcp.app.state.model_manager
        result = await model_manager.chat(
            model_id=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
        )
        return {"response": result.response, "metadata": result.metadata}
    
    @mcp.tool()
    async def load_model(
        model_id: str,
        device: str | None = None,
        num_gpu_layers: int | None = None,
        context_length: int | None = None,
        gpu_memory_utilization: float | None = None,
        max_model_len: int | None = None,
        quantization: str | None = None,
        trust_remote_code: bool = False
    ) -> dict[str, Any]:
        """Load a model into memory.
        
        Args:
            model_id: The ID of the model to load
            device: Device to load the model on (e.g., 'cuda', 'cpu', 'auto')
            num_gpu_layers: Number of GPU layers to use (for GPU acceleration)
            context_length: Maximum context length for the model
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
            max_model_len: Maximum sequence length for the model
            quantization: Quantization method (e.g., 'int4', 'int8', 'fp16')
            trust_remote_code: Whether to trust remote code execution (for custom models)
            
        Returns:
            Dictionary with model loading status and metadata
        """
        model_manager = mcp.app.state.model_manager
        try:
            # Create a dictionary of non-None parameters to pass to load_model
            load_kwargs = {
                k: v for k, v in {
                    'device': device,
                    'num_gpu_layers': num_gpu_layers,
                    'context_length': context_length,
                    'gpu_memory_utilization': gpu_memory_utilization,
                    'max_model_len': max_model_len,
                    'quantization': quantization,
                    'trust_remote_code': trust_remote_code
                }.items() if v is not None
            }
            
            model = await model_manager.load_model(model_id, **load_kwargs)
            return {
                "success": True,
                "model_id": model_id,
                "status": model.status.value if hasattr(model, 'status') else "loaded",
                "provider": model.provider.value if hasattr(model, 'provider') else "unknown"
            }
        except Exception as e:
            return {
                "success": False,
                "model_id": model_id,
                "error": str(e)
            }
    
    @mcp.tool()
    async def unload_model(model_id: str) -> dict[str, Any]:
        """Unload a model from memory.
        
        Args:
            model_id: The ID of the model to unload
            
        Returns:
            Dictionary with model unloading status
        """
        model_manager = mcp.app.state.model_manager
        try:
            success = await model_manager.unload_model(model_id)
            return {
                "success": success,
                "model_id": model_id,
                "message": f"Model {model_id} unloaded successfully" if success else f"Failed to unload model {model_id}"
            }
        except Exception as e:
            return {
                "success": False,
                "model_id": model_id,
                "error": str(e)
            }
    
    @mcp.tool()
    async def get_loaded_models() -> list[dict[str, Any]]:
        """Get a list of all currently loaded models.
        
        Returns:
            List of loaded models with their details
        """
        model_manager = mcp.app.state.model_manager
        try:
            # Get all models and filter for loaded ones
            all_models = await model_manager.list_models()
            loaded_models = [
                {
                    "id": model.id,
                    "name": model.name,
                    "provider": model.provider.value,
                    "status": model.status.value if hasattr(model, 'status') else "unknown"
                }
                for model in all_models
                if hasattr(model, 'status') and model.status == ModelStatus.LOADED
            ]
            return loaded_models
        except Exception as e:
            return [{"error": f"Error getting loaded models: {str(e)}"}]
    
    @mcp.tool()
    async def get_provider_status(provider_name: str) -> dict[str, Any]:
        """Get the status of a provider.
        
        Args:
            provider_name: Name of the provider (e.g., 'ollama', 'anthropic')
            
        Returns:
            Dictionary with provider status information
        """
        model_manager = mcp.app.state.model_manager
        try:
            # Get the provider factory from the model manager
            provider_factory = model_manager.provider_factory
            
            # Find the provider type by name (case-insensitive)
            provider_type = None
            for pt in ModelProvider:
                if pt.value.lower() == provider_name.lower():
                    provider_type = pt
                    break
            
            if not provider_type:
                return {
                    "provider": provider_name,
                    "status": "error",
                    "error": f"Provider '{provider_name}' not found"
                }
            
            # Check if provider is already initialized
            provider = provider_factory._providers.get(provider_type)
            
            if provider:
                # Provider is already loaded, check if it's ready
                is_ready = await provider.is_ready() if hasattr(provider, 'is_ready') else True
                return {
                    "provider": provider_name,
                    "status": "ready" if is_ready else "not_ready",
                    "initialized": True,
                    "details": {
                        "model_count": len(await provider.list_models()) if hasattr(provider, 'list_models') else "unknown"
                    }
                }
            else:
                # Provider not yet loaded
                return {
                    "provider": provider_name,
                    "status": "not_initialized",
                    "initialized": False
                }
                
        except Exception as e:
            return {
                "provider": provider_name,
                "status": "error",
                "error": str(e)
            }
    
    @mcp.tool()
    async def load_provider(
        provider_name: str,
        auto_start: bool = True,
        wait_until_ready: bool = True,
        timeout: int = 30
    ) -> dict[str, Any]:
        """Load and initialize a provider.
        
        Args:
            provider_name: Name of the provider to load (e.g., 'ollama')
            auto_start: Whether to automatically start the provider if not running
            wait_until_ready: Whether to wait until the provider is ready
            timeout: Maximum time to wait for provider to be ready (seconds)
            
        Returns:
            Dictionary with provider loading status and details
        """
        model_manager = mcp.app.state.model_manager
        provider_factory = model_manager.provider_factory
        
        try:
            # Find the provider type by name (case-insensitive)
            provider_type = None
            for pt in ModelProvider:
                if pt.value.lower() == provider_name.lower():
                    provider_type = pt
                    break
            
            if not provider_type:
                return {
                    "success": False,
                    "provider": provider_name,
                    "error": f"Provider '{provider_name}' not found"
                }
            
            # Check if provider is already loaded
            if provider_type in provider_factory._providers:
                return {
                    "success": True,
                    "provider": provider_name,
                    "status": "already_loaded",
                    "message": f"Provider '{provider_name}' is already loaded"
                }
            
            # Special handling for Ollama provider to auto-start if needed
            if provider_type == ModelProvider.OLLAMA and auto_start:
                try:
                    # Try to import the Ollama provider
                    from ..providers.ollama import OllamaProvider
                    
                    # Check if Ollama server is running
                    import httpx
                    from urllib.parse import urljoin
                    
                    # Get Ollama base URL from config or use default
                    ollama_config = model_manager.settings.providers.get('ollama', {})
                    base_url = ollama_config.get('base_url', 'http://localhost:11434')
                    
                    try:
                        async with httpx.AsyncClient() as client:
                            response = await client.get(urljoin(base_url, '/api/tags'), timeout=5.0)
                            response.raise_for_status()
                    except Exception as e:
                        # Ollama server not running, try to start it
                        import subprocess
                        import asyncio
                        
                        try:
                            # Try to start Ollama in the background
                            subprocess.Popen(['ollama', 'serve'])
                            
                            # Wait for server to start
                            if wait_until_ready:
                                start_time = asyncio.get_event_loop().time()
                                while (asyncio.get_event_loop().time() - start_time) < timeout:
                                    try:
                                        async with httpx.AsyncClient() as client:
                                            response = await client.get(
                                                urljoin(base_url, '/api/tags'), 
                                                timeout=2.0
                                            )
                                            if response.status_code == 200:
                                                break
                                    except:
                                        pass
                                    await asyncio.sleep(1)
                                
                                # Final check
                                async with httpx.AsyncClient() as client:
                                    response = await client.get(
                                        urljoin(base_url, '/api/tags'), 
                                        timeout=2.0
                                    )
                                    response.raise_for_status()
                            
                        except Exception as start_error:
                            return {
                                "success": False,
                                "provider": provider_name,
                                "error": f"Failed to start Ollama: {str(start_error)}",
                                "suggestion": "Make sure Ollama is installed and in your PATH"
                            }
                    
                except ImportError:
                    return {
                        "success": False,
                        "provider": provider_name,
                        "error": "Ollama provider not available",
                        "suggestion": "Install the Ollama provider with 'pip install ollama'",
                    }
            
            # Initialize the provider
            try:
                provider = provider_factory.get_provider(provider_type)
                
                # If provider has an async initialize method, call it
                if hasattr(provider, 'initialize') and callable(provider.initialize):
                    await provider.initialize()
                
                # Check if provider is ready
                is_ready = await provider.is_ready() if hasattr(provider, 'is_ready') else True
                
                if not is_ready and wait_until_ready:
                    # Wait for provider to be ready with timeout
                    start_time = asyncio.get_event_loop().time()
                    while not is_ready and (asyncio.get_event_loop().time() - start_time) < timeout:
                        await asyncio.sleep(1)
                        is_ready = await provider.is_ready() if hasattr(provider, 'is_ready') else True
                
                return {
                    "success": True,
                    "provider": provider_name,
                    "status": "ready" if is_ready else "not_ready",
                    "details": {
                        "model_count": len(await provider.list_models()) if hasattr(provider, 'list_models') else "unknown"
                    }
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "provider": provider_name,
                    "error": f"Failed to initialize provider: {str(e)}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "provider": provider_name,
                "error": str(e)
            }

def register_handlers(app: FastAPI, mcp: FastMCP) -> None:
    """Register startup and shutdown event handlers."""
    @app.on_event("startup")
    async def startup():
        await startup_event(app)
        setup_mcp(mcp)
    
    @app.on_event("shutdown")
    async def shutdown():
        await shutdown_event(app)
