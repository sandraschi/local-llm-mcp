""""API v1 endpoints package."""

# Import routers to make them available when importing from this package
from .models import router as models_router

# List of all routers to be included in the API
__all__ = ["models_router"]

# Re-export the router for easier imports
router = models_router
