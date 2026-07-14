"""API endpoints for system and GPU telemetry."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status

from ....utils.gpu import refresh_gpu_info

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


@router.get("/", response_model=dict[str, Any])
async def get_telemetry() -> dict[str, Any]:
    """Retrieve real-time system and GPU telemetry data.

    Returns:
        Dictionary containing GPU stats, system load, and memory info.
    """
    try:
        gpu_data = refresh_gpu_info()

        # In the future, we can add CPU/RAM stats here using psutil
        return {
            "gpu": gpu_data,
            "system": {
                "status": "online",
            },
        }
    except Exception as e:
        logger.error(f"Failed to retrieve telemetry: {e!s}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Telemetry error: {e!s}") from e
