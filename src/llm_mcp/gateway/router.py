"""FastAPI router for POST /v1/chat/completions (Lightport-compatible gateway)."""

import logging

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
        raise HTTPException(
            status_code=502,
            detail={"error": str(e), "provider": provider_name},
        )


@gateway_router.get("/v1/models")
async def list_models():
    """List all available models across registered gateway providers."""
    providers = list_providers()
    models = []
    for p in providers:
        models.append({
            "id": f"{p}/default",
            "object": "model",
            "created": 0,
            "owned_by": p,
        })
    return {"object": "list", "data": models}


@gateway_router.get("/v1/gateway/providers")
async def gateway_providers():
    """List registered gateway providers."""
    return {"providers": list_providers()}
