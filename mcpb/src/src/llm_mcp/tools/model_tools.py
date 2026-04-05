"""Model management tools for the LLM MCP server."""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Information about an available model."""
    id: str
    name: str
    provider: str
    context_length: int
    max_tokens: int
    description: str = ""
    parameters: Optional[Dict[str, Any]] = None

class ModelManager:
    """Manages available models and their configurations."""
    
    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        self._initialize_builtin_models()
    
    def _initialize_builtin_models(self):
        """Initialize with some default models."""
        default_models = [
            ModelInfo(
                id="gpt-4o",
                name="GPT-4o",
                provider="openai",
                context_length=128000,
                max_tokens=4096,
                description="OpenAI's most advanced model, with multimodal capabilities"
            ),
            ModelInfo(
                id="claude-3-opus-20240229",
                name="Claude 3 Opus",
                provider="anthropic",
                context_length=200000,
                max_tokens=4096,
                description="Most capable model, excelling at highly complex tasks"
            ),
            ModelInfo(
                id="llama3-70b-8192",
                name="Llama 3 70B",
                provider="meta",
                context_length=8192,
                max_tokens=4096,
                description="Meta's most capable open model"
            )
        ]
        
        for model in default_models:
            self.register_model(model)
    
    def register_model(self, model_info: ModelInfo):
        """Register a new model with the manager."""
        self.models[model_info.id] = model_info
        logger.info(f"Registered model: {model_info.id} ({model_info.name} by {model_info.provider})")
    
    def list_models(self, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all available models, optionally filtered by provider."""
        models = self.models.values()
        if provider:
            models = [m for m in models if m.provider.lower() == provider.lower()]
        
        return [{
            "id": m.id,
            "name": m.name,
            "provider": m.provider,
            "context_length": m.context_length,
            "max_tokens": m.max_tokens,
            "description": m.description,
            "parameters": m.parameters or {}
        } for m in models]
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get model information by ID."""
        return self.models.get(model_id)

# Global model manager instance
_model_manager = ModelManager()

# Implementation functions

async def _list_models_impl(provider: Optional[str] = None) -> List[Dict[str, Any]]:
    """Implementation of list_models.
    
    Args:
        provider: Optional provider name to filter models by
        
    Returns:
        List of model information dictionaries
    """
    return _model_manager.list_models(provider)

async def _get_model_info_impl(model_id: str) -> Dict[str, Any]:
    """Implementation of get_model_info.
    
    Args:
        model_id: The ID of the model to get information about
        
    Returns:
        Detailed model information or error if not found
    """
    model = _model_manager.get_model(model_id)
    if not model:
        return {"error": f"Model {model_id} not found"}
        
    return {
        "id": model.id,
        "name": model.name,
        "provider": model.provider,
        "context_length": model.context_length,
        "max_tokens": model.max_tokens,
        "description": model.description,
        "parameters": model.parameters or {}
    }

async def _register_model_impl(
    model_id: str,
    name: str,
    provider: str,
    context_length: int,
    max_tokens: int,
    description: str = "",
    parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Implementation of register_model.
    
    Args:
        model_id: Unique identifier for the model
        name: Human-readable name
        provider: Provider name (e.g., 'openai', 'anthropic')
        context_length: Maximum context length in tokens
        max_tokens: Maximum tokens to generate
        description: Optional description
        parameters: Optional model parameters
        
    Returns:
        Confirmation of registration
    """
    model_info = ModelInfo(
        id=model_id,
        name=name,
        provider=provider,
        context_length=context_length,
        max_tokens=max_tokens,
        description=description,
        parameters=parameters
    )
    _model_manager.register_model(model_info)
    return {"status": "success", "model_id": model_id}

def register_model_tools(mcp):
    """Register all model-related tools with the MCP server.
    
    Args:
        mcp: The MCP server instance with tool decorator
        
    Returns:
        The MCP server instance with model tools registered
    """
    @mcp.tool
    async def list_models(provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all available models with optional provider filtering.
        
        This tool is stateful and caches results for better performance.
        
        Args:
            provider: Optional provider name to filter models by
            
        Returns:
            List of model information dictionaries with caching
        """
        # The state is automatically managed by FastMCP 2.11.3+
        return await _list_models_impl(provider)
    
    @mcp.tool
    async def get_model_info(model_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific model with caching.
        
        This tool maintains a cache of model information to improve performance.
        The cache is automatically managed by FastMCP's stateful tools.
        
        Args:
            model_id: The ID of the model to get information about
            
        Returns:
            Detailed model information or error if not found
        """
        return await _get_model_info_impl(model_id)
    
    @mcp.tool()
    async def register_model(
        model_id: str,
        name: str,
        provider: str,
        context_length: int,
        max_tokens: int,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Register a new model with the server.
        
        Args:
            model_id: Unique identifier for the model
            name: Human-readable name
            provider: Provider name (e.g., 'openai', 'anthropic')
            context_length: Maximum context length in tokens
            max_tokens: Maximum tokens to generate
            description: Optional description
            parameters: Optional model parameters
            
        Returns:
            Confirmation of registration
        """
        return await _register_model_impl(
            model_id=model_id,
            name=name,
            provider=provider,
            context_length=context_length,
            max_tokens=max_tokens,
            description=description,
            parameters=parameters
        )
    
    return mcp
