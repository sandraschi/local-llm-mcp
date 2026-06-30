"""Lightport-compatible AI gateway for local-llm-mcp.

Provides POST /v1/chat/completions that translates OpenAI-format requests
to/from 10+ LLM provider native formats. Ported from glama-ai/lightport.
"""

from llm_mcp.gateway.router import gateway_router

__all__ = ["gateway_router"]
