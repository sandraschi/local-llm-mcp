"""Base models for LLM MCP Server."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ModelProvider(StrEnum):
    """Supported model providers."""

    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    VLLM = "vllm"
    OPENAI = "openai"
    GEMINI = "gemini"
    PERPLEXITY = "perplexity"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"


class ModelStatus(StrEnum):
    """Model status."""

    LOADED = "loaded"
    UNLOADED = "unloaded"
    LOADING = "loading"
    ERROR = "error"


class ModelCapability(StrEnum):
    """Model capabilities."""

    TEXT_GENERATION = "text_generation"
    CHAT = "chat"
    EMBEDDINGS = "embeddings"
    VISION = "vision"
    AUDIO = "audio"
    VIDEO_GENERATION = "video_generation"


class ModelMetadata(BaseModel):
    """Metadata for a model."""

    id: str = Field(..., description="Unique identifier for the model")
    name: str = Field(..., description="Display name of the model")
    provider: ModelProvider = Field(..., description="Provider of the model")
    version: str = Field(default="latest", description="Model version")
    status: ModelStatus = Field(
        default=ModelStatus.UNLOADED,
        description="Current status of the model",
    )
    capabilities: list[ModelCapability] = Field(
        default_factory=list,
        description="Capabilities supported by the model",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Model-specific parameters",
    )
    created_at: str | None = Field(
        default=None,
        description="Timestamp when the model was created",
    )
    updated_at: str | None = Field(
        default=None,
        description="Timestamp when the model was last updated",
    )


class BaseProvider(ABC):
    """Unified base class for LLM providers.

    Merges model lifecycle (models/base.py) with service provider interface
    (providers/base.py).  Methods marked abstract MUST be implemented;
    concrete defaults are provided where a sensible fallback exists.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config

    # ── Identity ──────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g. 'openai', 'ollama', 'huggingface')."""

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def is_ready(self) -> bool:
        return True

    # ── Model discovery ───────────────────────────────────────────────────

    @abstractmethod
    async def list_models(self) -> list[dict[str, Any]]:
        """List available models - each dict must include at least ``id``."""

    async def get_model(self, model_id: str) -> dict[str, Any] | None:
        """Resolve a single model by id via linear scan of ``list_models``."""
        for m in await self.list_models():
            if m.get("id") == model_id:
                return m
        return None

    async def get_model_info(self, model_name: str) -> dict[str, Any]:
        """Alias kept for backward compat; delegates to ``get_model``."""
        result = await self.get_model(model_name)
        return result or {}

    # ── Model lifecycle (override for stateful providers) ─────────────────

    async def load_model(self, model_id: str, **kwargs) -> dict[str, Any]:
        raise NotImplementedError(f"{type(self).__name__} does not support load_model")

    async def unload_model(self, model_id: str) -> bool:
        return False

    async def pull_model(self, model_name: str) -> dict[str, Any]:
        raise NotImplementedError(f"{type(self).__name__} does not support pull_model")

    # ── Text generation ───────────────────────────────────────────────────

    @abstractmethod
    def generate(self, prompt: str, model: str, **kwargs) -> AsyncGenerator[str, None]:
        """Streaming generation - the primitive every provider must supply.

        Implementations are async generator functions (``async def`` with
        ``yield``); the base is intentionally non-async so the declared type
        matches the AsyncGenerator contract instead of a coroutine wrapping it.
        """

    async def generate_text(self, model_id: str, prompt: str, **kwargs) -> str:
        """Non-streaming collect from ``generate``."""
        chunks: list[str] = []
        async for chunk in self.generate(prompt, model_id, **kwargs):  # ty: ignore[not-iterable]
            chunks.append(chunk)
        return "".join(chunks)

    async def chat(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> str:
        """Default: grab the last user message and pass to ``generate_text``."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return await self.generate_text(model_id, msg.get("content", ""), **kwargs)
        return ""

    async def generate_embeddings(self, model_id: str, texts: list[str], **kwargs) -> list[list[float]]:
        raise NotImplementedError(f"{type(self).__name__} does not support generate_embeddings")
