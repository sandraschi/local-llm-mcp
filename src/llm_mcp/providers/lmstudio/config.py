"""Configuration for LM Studio provider (0.4.0+)."""

import os
from typing import Any

from pydantic import BaseModel, Field


class LMStudioConfig(BaseModel):
    """Configuration for LM Studio provider.

    Attributes:
        base_url: Base URL for the LM Studio API (default: http://localhost:1234).
            Native REST lives at <base>/api/v1, OpenAI-compat at <base>/v1.
        timeout: Request timeout in seconds (default: 60).
        api_key: Permission key for Server auth (Settings > Server).
            Falls back to LM_API_TOKEN / LMSTUDIO_API_KEY env. Empty = no auth.
    """

    base_url: str = Field(default="http://localhost:1234", description="Base URL for the LM Studio API")
    timeout: int = Field(default=60, description="Request timeout in seconds", ge=1, le=300)
    api_key: str = Field(default="", description="Permission key (or LM_API_TOKEN env)")

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "LMStudioConfig":
        """Create an LMStudioConfig from a dictionary.

        Args:
            config_dict: Dictionary containing configuration

        Returns:
            LMStudioConfig instance
        """
        data = dict(config_dict)
        if not data.get("api_key"):
            data["api_key"] = os.getenv("LM_API_TOKEN", "") or os.getenv("LMSTUDIO_API_KEY", "")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Convert the config to a dictionary.

        Returns:
            Dictionary representation of the config (api_key redacted when empty only;
            callers must never log the key value).
        """
        d = self.model_dump(exclude_unset=True, exclude_none=True)
        if not d.get("api_key"):
            d.pop("api_key", None)
        return d

    @property
    def auth_headers(self) -> dict[str, str]:
        """Auth headers for LM Studio Server (empty when no key configured)."""
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}
