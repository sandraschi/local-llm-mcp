"""Text generation tools for the LLM MCP server."""
from typing import Any, Dict, List, Optional, Union, Literal
import logging
from dataclasses import dataclass, asdict
import time
import json

from .model_tools import _model_manager as model_manager, ModelInfo

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    top_k: Optional[int] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False

@dataclass
class ChatMessage:
    """A message in a chat conversation."""
    role: Literal["system", "user", "assistant", "function"]
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict] = None

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
        
    async def generate(
        self,
        model_id: str,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text using the specified model.
        
        Args:
            model_id: ID of the model to use
            prompt: Input prompt text
            config: Generation configuration
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Dictionary containing the generated text and metadata
        """
        model = model_manager.get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
            
        config = config or GenerationConfig()
        
        # Simulate generation (replace with actual implementation)
        start_time = time.time()
        
        # This is a placeholder - implement actual model calls here
        generated_text = f"Generated response for model {model_id} with prompt: {prompt[:50]}..."
        
        end_time = time.time()
        
        return {
            "text": generated_text,
            "model": model_id,
            "tokens_used": len(generated_text.split()),  # Rough estimate
            "time_taken": end_time - start_time,
            "finish_reason": "length"
        }
    
    async def chat(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a chat completion.
        
        Args:
            model_id: ID of the model to use
            messages: List of message dictionaries with 'role' and 'content'
            config: Generation configuration
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Dictionary containing the generated response and metadata
        """
        model = model_manager.get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
            
        config = config or GenerationConfig()
        
        # Simulate chat completion (replace with actual implementation)
        start_time = time.time()
        
        # This is a placeholder - implement actual chat completion here
        last_message = messages[-1]["content"] if messages else ""
        response_text = f"Chat response from {model_id} to: {last_message[:50]}..."
        
        end_time = time.time()
        
        return {
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "model": model_id,
            "tokens_used": len(response_text.split()),  # Rough estimate
            "time_taken": end_time - start_time,
            "finish_reason": "stop"
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
    **kwargs
) -> Dict[str, Any]:
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
    config = GenerationConfig(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=stream
    )
    
    return await generation_manager.generate(model, prompt, config, **kwargs)

async def chat_completion_impl(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 1000,
    top_p: float = 1.0,
    stream: bool = False,
    **kwargs
) -> Dict[str, Any]:
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
    config = GenerationConfig(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=stream
    )
    
    return await generation_manager.chat(model, messages, config, **kwargs)

async def embed_text_impl(
    model: str,
    text: Union[str, List[str]],
    **kwargs
) -> Dict[str, Any]:
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
        "data": [
            {
                "embedding": emb,
                "index": i,
                "object": "embedding"
            }
            for i, emb in enumerate(embeddings)
        ],
        "usage": {
            "prompt_tokens": sum(len(t.split()) for t in text),
            "total_tokens": sum(len(t.split()) for t in text)
        }
    }

def register_generation_tools(mcp):
    """Register all generation-related tools with the MCP server using FastMCP 2.11.3 stateful features.
    
    Args:
        mcp: The MCP server instance with tool decorator
        
    Returns:
        The MCP server instance with generation tools registered
        
    Notes:
        - Tools are registered with stateful=True to maintain state between invocations
        - State TTL is set based on the expected cache duration for each tool
    """
    tool = mcp.tool
    
    @tool(stateful=True, state_ttl=300)  # 5-minute cache for text generation
    async def generate_text(
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 1.0,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text using the specified model with stateful caching.
        
        This tool maintains a cache of recent generations to improve performance.
        The cache is automatically managed by FastMCP's stateful tools.
        
        Args:
            model: ID of the model to use
            prompt: Input prompt text
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter
            stream: Whether to stream the response
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing the generated text and metadata with caching
        """
        return await generate_text_impl(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=stream,
            **kwargs
        )
        
    @tool(stateful=True, state_ttl=600)  # 10-minute cache for chat completions
    async def chat_completion(
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 1.0,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
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
            **kwargs
        )
        
    @tool(stateful=True, state_ttl=86400)  # 24-hour cache for embeddings
    async def embed_text(
        model: str,
        text: Union[str, List[str]],
        **kwargs
    ) -> Dict[str, Any]:
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
        return await embed_text_impl(model=model, text=text, **kwargs)
        
    return mcp
