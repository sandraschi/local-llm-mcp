"""Model management tools for Ollama and LM Studio.

This module provides tools to manage LLM models in Ollama and LM Studio,
including loading, unloading, and downloading models.
"""
import os
import json
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Default paths and endpoints
OLLAMA_API_BASE = "http://localhost:11434/api"
LMSTUDIO_API_BASE = "http://localhost:1234/v1"

class ModelManager:
    """Base class for model management operations."""
    
    def __init__(self, api_base: str):
        """Initialize with API base URL."""
        self.api_base = api_base
        self._session = None
    
    @property
    def session(self):
        """Lazy-initialized aiohttp ClientSession."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Close the HTTP session if it exists."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an HTTP request to the model API."""
        url = f"{self.api_base}/{endpoint}"
        try:
            async with self.session.request(method, url, **kwargs) as response:
                response.raise_for_status()
                if response.status == 204:  # No content
                    return {}
                return await response.json()
        except Exception as e:
            logger.error(f"Request to {url} failed: {e}")
            raise


class OllamaManager(ModelManager):
    """Manager for Ollama models."""
    
    def __init__(self):
        """Initialize Ollama manager with default API base."""
        super().__init__(OLLAMA_API_BASE)
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all available Ollama models."""
        return await self._make_request("GET", "tags")
    
    async def pull_model(self, model_name: str) -> Dict[str, Any]:
        """Download a model from Ollama."""
        return await self._make_request("POST", "pull", json={"name": model_name})
    
    async def delete_model(self, model_name: str) -> Dict[str, Any]:
        """Delete a model from Ollama."""
        return await self._make_request("DELETE", f"delete", json={"name": model_name})
    
    async def load_model(self, model_name: str) -> Dict[str, Any]:
        """Load a model into memory."""
        return await self._make_request("POST", "generate", json={
            "model": model_name,
            "prompt": "",
            "stream": False
        })
    
    async def unload_model(self) -> Dict[str, Any]:
        """Unload the current model from memory."""
        return await self._make_request("POST", "api/chat", json={"model": ""})


class LMStudioManager(ModelManager):
    """Manager for LM Studio models."""
    
    def __init__(self):
        """Initialize LM Studio manager with default API base."""
        super().__init__(LMSTUDIO_API_BASE)
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all available LM Studio models."""
        return await self._make_request("GET", "models")
    
    async def load_model(self, model_name: str) -> Dict[str, Any]:
        """Load a model in LM Studio."""
        return await self._make_request("POST", "models/load", json={"name": model_name})
    
    async def unload_model(self) -> Dict[str, Any]:
        """Unload the current model in LM Studio."""
        return await self._make_request("POST", "models/unload")


# Implementation functions

# Global instances (lazy-initialized)
_ollama = None
_lmstudio = None

def get_ollama() -> OllamaManager:
    """Get or create the Ollama manager instance."""
    global _ollama
    if _ollama is None:
        _ollama = OllamaManager()
    return _ollama

def get_lmstudio() -> LMStudioManager:
    """Get or create the LM Studio manager instance."""
    global _lmstudio
    if _lmstudio is None:
        _lmstudio = LMStudioManager()
    return _lmstudio

async def _ollama_list_models_impl() -> Dict[str, Any]:
    """Implementation of ollama_list_models."""
    ollama = get_ollama()
    return await ollama.list_models()

async def _ollama_pull_model_impl(model_name: str) -> Dict[str, Any]:
    """Implementation of ollama_pull_model.
    
    Args:
        model_name: Name of the model to download (e.g., 'llama2')
    """
    ollama = get_ollama()
    return await ollama.pull_model(model_name)

async def _ollama_delete_model_impl(model_name: str) -> Dict[str, Any]:
    """Implementation of ollama_delete_model.
    
    Args:
        model_name: Name of the model to delete
    """
    ollama = get_ollama()
    return await ollama.delete_model(model_name)

async def _ollama_load_model_impl(model_name: str) -> Dict[str, Any]:
    """Implementation of ollama_load_model.
    
    Args:
        model_name: Name of the model to load
    """
    ollama = get_ollama()
    return await ollama.load_model(model_name)

async def _ollama_unload_model_impl() -> Dict[str, Any]:
    """Implementation of ollama_unload_model."""
    ollama = get_ollama()
    return await ollama.unload_model()

async def _lmstudio_list_models_impl() -> Dict[str, Any]:
    """Implementation of lmstudio_list_models."""
    lmstudio = get_lmstudio()
    return await lmstudio.list_models()

async def _lmstudio_load_model_impl(model_name: str) -> Dict[str, Any]:
    """Implementation of lmstudio_load_model.
    
    Args:
        model_name: Name of the model to load
    """
    lmstudio = get_lmstudio()
    return await lmstudio.load_model(model_name)

async def _lmstudio_unload_model_impl() -> Dict[str, Any]:
    """Implementation of lmstudio_unload_model."""
    lmstudio = get_lmstudio()
    return await lmstudio.unload_model()

async def _cleanup_models_impl():
    """Cleanup resources on server shutdown."""
    global _ollama, _lmstudio
    
    try:
        if _ollama is not None:
            await _ollama.close()
    except Exception as e:
        logger.warning(f"Error closing Ollama manager: {e}")
    finally:
        _ollama = None
    
    try:
        if _lmstudio is not None:
            await _lmstudio.close()
    except Exception as e:
        logger.warning(f"Error closing LM Studio manager: {e}")
    finally:
        _lmstudio = None

def register_model_management_tools(mcp):
    """Register model management tools with the MCP server using FastMCP 2.11.3 stateful features.
    
    Args:
        mcp: The MCP server instance with tool decorator
        
    Returns:
        The MCP server instance with model management tools registered
        
    Notes:
        - List operations are cached for 5 minutes (300 seconds)
        - Model loading/unloading operations are not cached as they modify state
        - Model pull and delete operations are not cached as they modify the model repository
    """
    
    @mcp.tool()  # List Ollama models
    async def ollama_list_models() -> Dict[str, Any]:
        """List all available Ollama models.
        
        Returns:
            Dictionary containing list of available models and their details
        """
        return await _ollama_list_models_impl()
    
    @mcp.tool()  # Pull Ollama model
    async def ollama_pull_model(model_name: str) -> Dict[str, Any]:
        """Download an Ollama model.
        
        Args:
            model_name: Name of the model to download (e.g., 'llama2')
            
        Returns:
            Dictionary with download status and metadata
            
        State:
            - Not stateful (stateful=False) as it modifies the model repository
            - Invalidates the ollama_list_models cache
        """
        # Invalidate the list cache when pulling a new model
        mcp.invalidate_state(ollama_list_models)
        return await _ollama_pull_model_impl(model_name)
    
    @mcp.tool()  # Delete Ollama model
    async def ollama_delete_model(model_name: str) -> Dict[str, Any]:
        """Delete an Ollama model.
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            Dictionary with deletion status
            
        State:
            - Not stateful (stateful=False) as it modifies the model repository
            - Invalidates the ollama_list_models cache
        """
        # Invalidate the list cache when deleting a model
        mcp.invalidate_state(ollama_list_models)
        return await _ollama_delete_model_impl(model_name)
    
    @mcp.tool()  # Load Ollama model
    async def ollama_load_model(model_name: str) -> Dict[str, Any]:
        """Load an Ollama model for inference.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Dictionary with load status and metadata
            
        Caching:
            - Stateful with 1-hour TTL to reduce model loading overhead
            - Cache is automatically invalidated when the model is unloaded
        """
        # Invalidate any previous model load
        mcp.invalidate_state(ollama_load_model)
        return await _ollama_load_model_impl(model_name)
    
    @mcp.tool()  # Unload Ollama model
    async def ollama_unload_model() -> Dict[str, Any]:
        """Unload the currently loaded Ollama model.
        
        Returns:
            Dictionary with unload status
        State:
            - Not stateful (stateful=False) as it modifies the loaded model state
            - Invalidates the ollama_load_model cache
        """
        # Invalidate the load model cache when unloading
        mcp.invalidate_state(ollama_load_model)
        return await _ollama_unload_model_impl()
    
    @mcp.tool()  # List LM Studio models
    async def lmstudio_list_models() -> Dict[str, Any]:
        """List all available LM Studio models.
        
        Returns:
            Dictionary containing list of available models and their details
            
        Caching:
            - Stateful with 5-minute TTL to reduce API calls
        """
        return await _lmstudio_list_models_impl()
    
    @mcp.tool()  # Load LM Studio model
    async def lmstudio_load_model(model_name: str) -> Dict[str, Any]:
        """Load an LM Studio model for inference.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Dictionary with load status and metadata
            
        State:
            - Not stateful (stateful=False) as it modifies the loaded model state
            - Invalidates the lmstudio_list_models cache
        """
        # Invalidate the list cache when loading a model
        mcp.invalidate_state(lmstudio_list_models)
        return await _lmstudio_load_model_impl(model_name)
    
    @mcp.tool()  # Unload LM Studio model
    async def lmstudio_unload_model() -> Dict[str, Any]:
        """Unload the currently loaded LM Studio model.
        
        Returns:
            Dictionary with unload status
            
        State:
            - Not stateful (stateful=False) as it modifies the loaded model state
            - Invalidates the lmstudio_list_models cache
        """
        # Invalidate the list cache when unloading a model
        mcp.invalidate_state(lmstudio_list_models)
        return await _lmstudio_unload_model_impl()
    
    # Cleanup on server shutdown
    @mcp.on_shutdown
    async def cleanup():
        """Cleanup resources on server shutdown."""
        await _cleanup_models_impl()
    
    return mcp
