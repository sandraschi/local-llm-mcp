"""Model management tools for Ollama and LM Studio.

This module provides tools to manage LLM models in Ollama and LM Studio,
including loading, unloading, and downloading models. All operations route
through the unified ProviderHealthService for liveness checks and circuit
breaking before making real API calls.
"""

import asyncio
import logging
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

# Default paths and endpoints
OLLAMA_API_BASE = "http://localhost:11434/api"
LMSTUDIO_API_BASE = "http://localhost:1234/v1"

# Granular timeouts: fast connect, longer read for inference
_CONNECT_TIMEOUT = aiohttp.ClientTimeout(total=30, connect=5)
_PULL_TIMEOUT = aiohttp.ClientTimeout(total=300, connect=10)


def _invalidate_health(provider_name: str) -> None:
    """Clear cached health for a provider to force re-check on next request."""
    try:
        from llm_mcp.services.provider_health import invalidate_provider_health

        invalidate_provider_health(provider_name)
    except Exception:
        logger.debug("Could not invalidate provider health cache", exc_info=True)


class ModelManager:
    """Base class for model management operations with health-check integration."""

    def __init__(self, api_base: str, provider_name: str):
        """Initialize with API base URL and provider name for health routing."""
        self.api_base = api_base
        self.provider_name = provider_name
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

    async def _check_health(self) -> dict[str, Any]:
        """Verify provider liveness before making a real request.

        Returns health dict with 'reachable' bool. Raises ConnectionError if
        circuit breaker is open or health check fails.
        """
        from llm_mcp.services.provider_health import (
            check_lmstudio_health,
            check_ollama_health,
        )

        if self.provider_name == "ollama":
            health = await check_ollama_health()
        elif self.provider_name == "lmstudio":
            health = await check_lmstudio_health()
        else:
            return {"reachable": True}

        if not health.reachable:
            raise ConnectionError(
                f"{self.provider_name} is unreachable: {health.error}. Suggestion: {health.suggestion}"
            )
        return {"reachable": True, "latency_ms": health.latency_ms}

    async def _make_request(self, method: str, endpoint: str, **kwargs) -> dict[str, Any]:
        """Make an HTTP request with health check, retry, and structured errors."""
        await self._check_health()

        url = f"{self.api_base}/{endpoint}"
        timeout_override = kwargs.pop("_timeout", _CONNECT_TIMEOUT)
        kwargs.setdefault("timeout", timeout_override)

        last_error = None
        for attempt in range(3):
            try:
                async with self.session.request(method, url, **kwargs) as response:
                    response.raise_for_status()
                    if response.status == 204:
                        return {}
                    return await response.json()
            except aiohttp.ClientConnectorError as e:
                last_error = e
                if attempt < 2:
                    delay = (attempt + 1) * 1.0
                    logger.warning(
                        "Connection to %s failed (attempt %d/3), retrying in %.1fs: %s",
                        self.provider_name,
                        attempt + 1,
                        delay,
                        e,
                    )
                    await asyncio.sleep(delay)
                    # Force health re-check on retry
                    _invalidate_health(self.provider_name)
                else:
                    raise ConnectionError(
                        f"{self.provider_name} is not reachable after 3 attempts. "
                        f"Last error: {e}. Check that the service is running."
                    ) from e
            except aiohttp.ClientResponseError as e:
                raise ConnectionError(f"{self.provider_name} API error: HTTP {e.status} — {e.message}") from e
            except TimeoutError as e:
                raise TimeoutError(f"{self.provider_name} request timed out. The daemon may be hung.") from e
            except Exception as e:
                logger.error("Request to %s failed: %s", url, e)
                raise

        raise ConnectionError(f"{self.provider_name}: max retries exceeded") from last_error


class OllamaManager(ModelManager):
    """Manager for Ollama models with correct API semantics."""

    def __init__(self):
        """Initialize Ollama manager with default API base."""
        super().__init__(OLLAMA_API_BASE, "ollama")

    async def list_models(self) -> dict[str, Any]:
        """List all available Ollama models."""
        return await self._make_request("GET", "tags")

    async def pull_model(self, model_name: str) -> dict[str, Any]:
        """Download a model from Ollama (long timeout for large downloads)."""
        return await self._make_request("POST", "pull", json={"name": model_name}, _timeout=_PULL_TIMEOUT)

    async def delete_model(self, model_name: str) -> dict[str, Any]:
        """Delete a model from Ollama."""
        return await self._make_request("DELETE", "delete", json={"name": model_name})

    async def load_model(self, model_name: str) -> dict[str, Any]:
        """Load a model into memory and keep it alive.

        Uses /api/chat with keep_alive to trigger model warm-up
        without producing side effects.
        """
        return await self._make_request(
            "POST",
            "chat",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "ping"}],
                "stream": False,
                "keep_alive": -1,
            },
        )

    async def unload_model(self, model_name: str) -> dict[str, Any]:
        """Unload a model from memory using keep_alive=0.

        Sets keep_alive to 0 which causes Ollama to unload the model
        after the current request completes.
        """
        return await self._make_request(
            "POST",
            "generate",
            json={
                "model": model_name,
                "prompt": "",
                "stream": False,
                "keep_alive": 0,
            },
        )


class LMStudioManager(ModelManager):
    """Manager for LM Studio models."""

    def __init__(self):
        """Initialize LM Studio manager with default API base."""
        super().__init__(LMSTUDIO_API_BASE, "lmstudio")

    async def list_models(self) -> dict[str, Any]:
        """List all available LM Studio models."""
        return await self._make_request("GET", "models")

    async def load_model(self, model_name: str) -> dict[str, Any]:
        """Load a model in LM Studio."""
        return await self._make_request("POST", "models/load", json={"name": model_name})

    async def unload_model(self) -> dict[str, Any]:
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


async def _ollama_list_models_impl() -> dict[str, Any]:
    """Implementation of ollama_list_models."""
    ollama = get_ollama()
    return await ollama.list_models()


async def _ollama_pull_model_impl(model_name: str) -> dict[str, Any]:
    """Implementation of ollama_pull_model.

    Args:
        model_name: Name of the model to download (e.g., 'llama2')
    """
    ollama = get_ollama()
    return await ollama.pull_model(model_name)


async def _ollama_delete_model_impl(model_name: str) -> dict[str, Any]:
    """Implementation of ollama_delete_model.

    Args:
        model_name: Name of the model to delete
    """
    ollama = get_ollama()
    return await ollama.delete_model(model_name)


async def _ollama_load_model_impl(model_name: str) -> dict[str, Any]:
    """Implementation of ollama_load_model.

    Args:
        model_name: Name of the model to load
    """
    ollama = get_ollama()
    return await ollama.load_model(model_name)


async def _ollama_unload_model_impl(model_name: str) -> dict[str, Any]:
    """Implementation of ollama_unload_model.

    Uses keep_alive=0 to trigger immediate model unload from Ollama memory.
    """
    ollama = get_ollama()
    return await ollama.unload_model(model_name)


async def _lmstudio_list_models_impl() -> dict[str, Any]:
    """Implementation of lmstudio_list_models."""
    lmstudio = get_lmstudio()
    return await lmstudio.list_models()


async def _lmstudio_load_model_impl(model_name: str) -> dict[str, Any]:
    """Implementation of lmstudio_load_model.

    Args:
        model_name: Name of the model to load
    """
    lmstudio = get_lmstudio()
    return await lmstudio.load_model(model_name)


async def _lmstudio_unload_model_impl() -> dict[str, Any]:
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
    """Register all model management tools with the MCP server.

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
    async def ollama_list_models() -> dict[str, Any]:
        """List all available Ollama models.

        Returns:
            Dictionary containing list of available models and their details
        """
        return await _ollama_list_models_impl()

    @mcp.tool()  # Pull Ollama model
    async def ollama_pull_model(model_name: str) -> dict[str, Any]:
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
    async def ollama_delete_model(model_name: str) -> dict[str, Any]:
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
    async def ollama_load_model(model_name: str) -> dict[str, Any]:
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
    async def ollama_unload_model(model_name: str = "") -> dict[str, Any]:
        """Unload the currently loaded Ollama model.

        Args:
            model_name: Name of the model to unload from memory.

        Returns:
            Dictionary with unload status
        State:
            - Not stateful (stateful=False) as it modifies the loaded model state
            - Invalidates the ollama_load_model cache
        """
        # Invalidate the load model cache when unloading
        mcp.invalidate_state(ollama_load_model)
        return await _ollama_unload_model_impl(model_name)

    @mcp.tool()  # List LM Studio models
    async def lmstudio_list_models() -> dict[str, Any]:
        """List all available LM Studio models.

        Returns:
            Dictionary containing list of available models and their details

        Caching:
            - Stateful with 5-minute TTL to reduce API calls
        """
        return await _lmstudio_list_models_impl()

    @mcp.tool()  # Load LM Studio model
    async def lmstudio_load_model(model_name: str) -> dict[str, Any]:
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
    async def lmstudio_unload_model() -> dict[str, Any]:
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

    return mcp
