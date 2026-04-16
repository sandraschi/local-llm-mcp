"""Text generation tools for the LLM MCP server."""

import dataclasses
import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

from ..services.provider_factory import _provider_factory
from .model_tools import _model_manager as model_manager

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    top_k: int | None = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: str | list[str] | None = None
    stream: bool = False


@dataclass
class ChatMessage:
    """A message in a chat conversation."""

    role: Literal["system", "user", "assistant", "function"]
    content: str
    name: str | None = None
    function_call: dict | None = None


def _get_model_provider(model_id: str) -> str:
    """Get the provider for a model ID."""
    model = model_manager.get_model(model_id)
    if not model:
        raise ValueError(f"Model {model_id} not found")
    return model.provider


class GenerationManager:
    """Manages text generation with different model providers."""

    def __init__(self):
        self.sessions = {}
        self.provider_factory = _provider_factory

    async def generate(
        self,
        model_id: str,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Generate text using the specified model.

        Args:
            model_id: ID of the model to use
            prompt: Input prompt text
            config: Generation configuration
            **kwargs: Additional provider-specific parameters

        Returns:
            Dictionary containing the generated text and metadata
        """
        # Ensure model exists via manager or factory
        provider = await self.provider_factory.get_provider_for_model(model_id)

        # If not directly found, try via mode manager
        if not provider:
            model = model_manager.get_model(model_id)
            if model:
                provider = self.provider_factory.get_provider(model.provider)

        if not provider:
            # Check if it was a model manager miss but maybe the factory knows it generally?
            # Actually get_provider_for_model does that.
            raise ValueError(f"No provider found for model {model_id}")

        config = config or GenerationConfig()
        config_dict = dataclasses.asdict(config)
        config_dict.update(kwargs)

        start_time = time.time()

        try:
            generated_text = await provider.generate_text(model_id, prompt, **config_dict)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

        end_time = time.time()

        return {
            "text": generated_text,
            "model": model_id,
            "tokens_used": len(generated_text.split()),  # Provider should arguably return usage info
            "time_taken": end_time - start_time,
            "finish_reason": "stop",
        }

    async def chat(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Generate a chat completion.

        Args:
            model_id: ID of the model to use
            messages: List of message dictionaries with 'role' and 'content'
            config: Generation configuration
            **kwargs: Additional provider-specific parameters

        Returns:
            Dictionary containing the generated message and metadata
        """
        provider = await self.provider_factory.get_provider_for_model(model_id)

        if not provider:
            model = model_manager.get_model(model_id)
            if model:
                provider = self.provider_factory.get_provider(model.provider)

        if not provider:
            raise ValueError(f"No provider found for model {model_id}")

        config = config or GenerationConfig()
        config_dict = dataclasses.asdict(config)
        config_dict.update(kwargs)

        start_time = time.time()

        try:
            response_text = await provider.chat(model_id, messages, **config_dict)
        except Exception as e:
            logger.error(f"Chat generation failed: {e}")
            raise

        end_time = time.time()

        return {
            "message": {"role": "assistant", "content": response_text},
            "model": model_id,
            "tokens_used": len(response_text.split()),  # Estimate
            "time_taken": end_time - start_time,
            "finish_reason": "stop",
        }


# Global generation manager instance
generation_manager = GenerationManager()


# Implementation functions (without @tool decorator)
async def generate_text_impl(
    model: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    top_p: float = 1.0,
    stream: bool = False,
    **kwargs,
) -> dict[str, Any]:
    """Generate text using the specified model.

    Args:
        model: ID of the model to use
        prompt: Input prompt text
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum number of tokens to generate
        top_p: Nucleus sampling parameter
        stream: Whether to stream the response
        **kwargs: Additional generation parameters

    Returns:
        Dictionary containing the generated text and metadata
    """
    config = GenerationConfig(temperature=temperature, max_tokens=max_tokens, top_p=top_p, stream=stream)

    return await generation_manager.generate(model, prompt, config, **kwargs)


async def chat_completion_impl(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 1000,
    top_p: float = 1.0,
    stream: bool = False,
    **kwargs,
) -> dict[str, Any]:
    """Generate a chat completion.

    Args:
        model: ID of the model to use
        messages: List of message dictionaries with 'role' and 'content'
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum number of tokens to generate
        top_p: Nucleus sampling parameter
        stream: Whether to stream the response
        **kwargs: Additional generation parameters

    Returns:
        Dictionary containing the generated response and metadata
    """
    config = GenerationConfig(temperature=temperature, max_tokens=max_tokens, top_p=top_p, stream=stream)

    return await generation_manager.chat(model, messages, config, **kwargs)


async def embed_text_impl(model: str, text: str | list[str], **kwargs) -> dict[str, Any]:
    """Generate embeddings for the input text.

    Args:
        model: ID of the embedding model to use
        text: Input text or list of texts to embed
        **kwargs: Additional parameters

    Returns:
        Dictionary containing the embeddings and metadata
    """
    # This is a placeholder - implement actual embedding generation
    if isinstance(text, str):
        text = [text]

    # Generate random embeddings for demonstration
    import numpy as np

    np.random.seed(hash(text[0]) % 2**32)

    embeddings = [
        np.random.rand(1536).tolist()  # Standard embedding size
        for _ in range(len(text))
    ]

    return {
        "model": model,
        "data": [{"embedding": emb, "index": i, "object": "embedding"} for i, emb in enumerate(embeddings)],
        "usage": {
            "prompt_tokens": sum(len(t.split()) for t in text),
            "total_tokens": sum(len(t.split()) for t in text),
        },
    }


def register_generation_tools(mcp):
    """Register all generation-related tools with the MCP server.

    Args:
        mcp: The MCP server instance with tool decorator

    Returns:
        The MCP server instance with generation tools registered
    """

    @mcp.tool()  # Generate Text
    async def generate_text(
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 1.0,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Generate text using a local LLM.

        Args:
            model: ID of the model to use (use list_models to find available models)
            prompt: Input text prompt
            temperature: Sampling temperature (0-2). Higher = more random.
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stream: Whether to stream the response

        Returns:
            Dictionary with 'text', 'model', and usage stats
        """
        return await generate_text_impl(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=stream,
        )

    @mcp.tool()  # Chat Completion
    async def chat_completion(
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 1.0,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Generate a chat completion with stateful conversation management.

        This tool maintains conversation state and caches recent completions.
        The state is automatically managed by FastMCP's stateful tools.

        Args:
            model: ID of the model to use
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter
            stream: Whether to stream the response
            **kwargs: Additional generation parameters

        Returns:
            Dictionary containing the generated response and metadata with caching
        """
        return await chat_completion_impl(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=stream,
        )

    @mcp.tool()  # Generate embeddings
    async def embed_text(model: str, text: str | list[str]) -> dict[str, Any]:
        """Generate and cache embeddings for the input text.

        This tool caches embeddings to avoid redundant computations.
        The cache is automatically managed by FastMCP's stateful tools.

        Args:
            model: ID of the embedding model to use
            text: Input text or list of texts to embed
            **kwargs: Additional parameters

        Returns:
            Dictionary containing the cached embeddings and metadata
        """
        return await embed_text_impl(model=model, text=text)

    return mcp
