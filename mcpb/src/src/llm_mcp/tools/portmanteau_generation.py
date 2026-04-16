"""LLM Generation portmanteau tool for Local LLM MCP server.

This tool consolidates all text generation, chat completion, and embedding operations
into a single interface following the portmanteau pattern.
"""

from typing import Any

from llm_mcp.tools.generation_tools import chat_completion_impl, embed_text_impl, generate_text_impl
from llm_mcp.utils.logging import get_logger

logger = get_logger(__name__)

# Import FastMCP components
try:
    from fastmcp import FastMCP
    from fastmcp.tools import Tool
    FASTMCP_AVAILABLE = True
except ImportError:
    logger.error("FastMCP not available - portmanteau tools require FastMCP >= 2.12.0")
    FASTMCP_AVAILABLE = False


async def llm_generation(
    operation: str,
    # Text generation
    model: str | None = None,
    prompt: str | None = None,
    # Chat completion
    messages: list[dict[str, str]] | None = None,
    # Common parameters
    temperature: float = 0.7,
    max_tokens: int | None = None,
    top_p: float = 1.0,
    stream: bool = False,
    # Embeddings
    text: str | list[str] | None = None,
    # Additional parameters (explicit instead of **kwargs)
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    stop: list[str] | None = None
) -> dict[str, Any]:
    """Comprehensive text generation and chat tool for Local LLM MCP server.

    PORTMANTEAU PATTERN: Consolidates 3 core generation operations into one tool.

    SUPPORTED OPERATIONS:
    - generate_text: Generate text from prompt (requires model, prompt)
    - chat_completion: Generate chat completion from messages (requires model, messages)
    - embed_text: Generate embeddings from text (requires model, text)

    Args:
        operation: Operation to perform (generate_text, chat_completion, embed_text)
        model: Model ID to use for generation/embedding
        prompt: Text prompt for generate_text operation
        messages: Chat messages for chat_completion operation (list of {"role": "user/assistant", "content": "..."})
        temperature: Sampling temperature (0.0-2.0, default: 0.7)
        max_tokens: Maximum tokens to generate (optional)
        top_p: Nucleus sampling parameter (0.0-1.0, default: 1.0)
        stream: Whether to stream response (default: False)
        text: Text or list of texts for embed_text operation
        frequency_penalty: Frequency penalty parameter (default: 0.0)
        presence_penalty: Presence penalty parameter (default: 0.0)
        stop: List of stop sequences (optional)

    Returns:
        Operation-specific result dictionary with generated content and metadata
    """
    try:
        if operation == "generate_text":
            if not model or not prompt:
                return {"error": "model and prompt required for generate_text operation"}
            return await generate_text_impl(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=stream,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop
            )

        elif operation == "chat_completion":
            if not model or not messages:
                return {"error": "model and messages required for chat_completion operation"}
            return await chat_completion_impl(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=stream,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop
            )

        elif operation == "embed_text":
            if not model or not text:
                return {"error": "model and text required for embed_text operation"}
            return await embed_text_impl(
                model=model,
                text=text
            )

        else:
            return {
                "error": f"Unknown operation: {operation}",
                "available_operations": ["generate_text", "chat_completion", "embed_text"]
            }

    except Exception as e:
        logger.error(f"Error in llm_generation operation {operation}: {e}", exc_info=True)
        return {"error": f"Operation failed: {e!s}", "operation": operation}


def register_llm_generation_tools(mcp):
    """Register the LLM Generation portmanteau tool with the MCP server."""
    if not FASTMCP_AVAILABLE:
        logger.error("Cannot register LLM Generation tools - FastMCP not available")
        return mcp

    @mcp.tool()
    async def llm_generation_tool(
        operation: str,
        model: str | None = None,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        top_p: float = 1.0,
        stream: bool = False,
        text: str | list[str] | None = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None
    ) -> dict[str, Any]:
        """LLM Generation Portmanteau Tool - Consolidated text generation operations.

        This tool consolidates all text generation, chat completion, and embedding operations
        into a single interface, reducing the number of MCP tools while maintaining full functionality.

        Use the 'operation' parameter to specify what you want to do:
        - generate_text: Generate text from a prompt
        - chat_completion: Generate chat completions from messages
        - embed_text: Generate embeddings from text
        """
        return await llm_generation(
            operation=operation,
            model=model,
            prompt=prompt,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=stream,
            text=text,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop
        )

    logger.info("Registered LLM Generation portmanteau tool")
    return mcp
