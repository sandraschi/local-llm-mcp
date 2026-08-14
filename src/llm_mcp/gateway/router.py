"""FastAPI router for POST /v1/chat/completions (Lightport-compatible gateway)."""

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from llm_mcp.gateway.base import get_adapter, list_providers

logger = logging.getLogger(__name__)
gateway_router = APIRouter()


@gateway_router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint.

    Translates incoming OpenAI-format requests to the target provider's
    native format and back. Selects provider via:
    1. x-lightport-provider header (explicit)
    2. Model ID prefix matching (e.g. "anthropic/..." -> anthropic)

    Returns standard OpenAI ChatCompletion JSON.
    """
    body = await request.json()
    headers = dict(request.headers)

    # Determine provider
    provider_name = headers.get("x-lightport-provider", "")
    if not provider_name:
        model = body.get("model", "")
        provider_name = model.split("/")[0] if "/" in model else "openai"

    adapter = get_adapter(provider_name)
    if not adapter:
        available = list_providers()
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Unknown provider '{provider_name}'. Available: {', '.join(available)}",
                "provider": provider_name,
                "available_providers": available,
            },
        )

    try:
        result = await adapter.complete(body, headers)
        return result
    except Exception as e:
        logger.error("Gateway error for provider '%s': %s", provider_name, e)
        error_type = "gateway_error"
        suggestion = None

        err_str = str(e).lower()
        if "connection" in err_str or "connect" in err_str:
            error_type = "connection_error"
            suggestion = f"Check that the {provider_name} service is running and reachable"
        elif "timeout" in err_str:
            error_type = "timeout"
            suggestion = f"The {provider_name} service is not responding — it may be hung"
        elif "refused" in err_str:
            error_type = "connection_refused"
            suggestion = f"The {provider_name} service is not running"
        elif "403" in err_str or "401" in err_str:
            error_type = "auth_error"
            suggestion = f"Check API key or authentication for {provider_name}"

        raise HTTPException(
            status_code=502,
            detail={
                "error": str(e),
                "error_type": error_type,
                "provider": provider_name,
                "suggestion": suggestion,
            },
        ) from e


@gateway_router.get("/v1/models")
async def list_models():
    """List all available models across registered gateway providers."""
    providers = list_providers()
    models = []
    for p in providers:
        models.append(
            {
                "id": f"{p}/default",
                "object": "model",
                "created": 0,
                "owned_by": p,
            }
        )
    return {"object": "list", "data": models}


@gateway_router.get("/v1/gateway/providers")
async def gateway_providers():
    """List registered gateway providers."""
    return {"providers": list_providers()}


@gateway_router.get("/v1/gateway/providers/health")
async def gateway_provider_health(request: Request):
    """Probe health of all registered gateway providers.

    Returns per-provider reachability with latency and error details.
    Local providers (Ollama, LM Studio) use the unified health service;
    cloud providers report a fast URL connectivity check.
    """
    start = time.monotonic()
    to_dict: Any = None
    try:
        from llm_mcp.services.provider_health import (
            check_all_providers,
            provider_health_to_dict,
        )

        to_dict = provider_health_to_dict
        local_health = await check_all_providers(force=True)
    except Exception as e:
        local_health = {
            "ollama": {
                "provider": "ollama",
                "reachable": False,
                "error": str(e),
            },
            "lmstudio": {
                "provider": "lmstudio",
                "reachable": False,
                "error": str(e),
            },
        }

    results: dict[str, Any] = {}
    for name, h in local_health.items():
        results[name] = h if isinstance(h, dict) else to_dict(h)

    # Mark all registered providers (even non-local ones)
    for p in list_providers():
        if p not in results:
            results[p] = {"provider": p, "reachable": True, "note": "Cloud provider — not probed"}

    elapsed_ms = round((time.monotonic() - start) * 1000, 1)
    return {"providers": results, "elapsed_ms": elapsed_ms}
