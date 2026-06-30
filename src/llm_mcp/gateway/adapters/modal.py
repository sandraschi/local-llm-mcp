"""Modal adapter — OpenAI-compatible."""

from llm_mcp.gateway.adapters.openai import OpenAIAdapter
from llm_mcp.gateway.base import register_provider


@register_provider("modal")
class ModalAdapter(OpenAIAdapter):
    provider = "modal"
    base_url = ""  # User's Modal endpoint
    api_key_env = "MODAL_API_KEY"

    def get_base_url(self, headers: dict[str, str]) -> str:
        import os
        return os.getenv("MODAL_BASE_URL", "https://--your-app--.modal.run/v1")
