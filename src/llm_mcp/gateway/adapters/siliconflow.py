"""SiliconFlow adapter — OpenAI-compatible."""

from llm_mcp.gateway.adapters.openai import OpenAIAdapter
from llm_mcp.gateway.base import register_provider


@register_provider("siliconflow")
class SiliconFlowAdapter(OpenAIAdapter):
    provider = "siliconflow"
    base_url = "https://api.siliconflow.cn/v1"
    api_key_env = "SILICONFLOW_API_KEY"
