"""Configuration API endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException

from ....core.config import get_settings

router = APIRouter()


@router.get("")
async def get_config():
    """Get the current server configuration."""
    settings = get_settings()
    return settings.to_dict()


@router.patch("")
async def update_config(updates: dict[str, Any]):
    """Update the server configuration."""
    try:
        settings = get_settings()
        settings.update_and_save(updates)
        return {"status": "success", "message": "Configuration updated and persisted to .env"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
