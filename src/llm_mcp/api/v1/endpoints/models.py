"""API endpoints for model management."""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ....models.base import ModelMetadata, ModelProvider, ModelStatus
from ....services.model_manager import ModelManager
from ....core.config import get_settings

# Create router
router = APIRouter()

# Response models
class ModelResponse(BaseModel):
    """Response model for a single model."""
    id: str = Field(..., description="Unique identifier for the model")
    name: str = Field(..., description="Display name of the model")
    provider: str = Field(..., description="Provider of the model")
    version: str = Field(..., description="Model version")
    status: str = Field(..., description="Current status of the model")
    capabilities: List[str] = Field(..., description="Capabilities supported by the model")
    created_at: Optional[str] = Field(None, description="Timestamp when the model was created")
    updated_at: Optional[str] = Field(None, description="Timestamp when the model was last updated")

class ModelListResponse(BaseModel):
    """Response model for a list of models."""
    models: List[ModelResponse] = Field(..., description="List of models")

class LoadModelRequest(BaseModel):
    """Request model for loading a model."""
    model_id: str = Field(..., description="ID of the model to load")
    params: Optional[dict] = Field(
        default_factory=dict,
        description="Additional parameters for model loading"
    )

class GenerateTextRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="The prompt to generate text from")
    model_id: Optional[str] = Field(
        None,
        description="ID of the model to use (defaults to the first available model if not specified)"
    )
    params: Optional[dict] = Field(
        default_factory=dict,
        description="Additional parameters for text generation"
    )

class ChatMessage(BaseModel):
    """A message in a chat conversation."""
    role: str = Field(..., description="Role of the message sender (e.g., 'user', 'assistant')")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    """Request model for chat completion."""
    messages: List[ChatMessage] = Field(
        ...,
        description="List of messages in the conversation"
    )
    model_id: Optional[str] = Field(
        None,
        description="ID of the model to use (defaults to the first available model if not specified)"
    )
    params: Optional[dict] = Field(
        default_factory=dict,
        description="Additional parameters for chat completion"
    )

# Helper function to convert internal model to API response
def model_to_response(model: ModelMetadata) -> ModelResponse:
    """Convert a ModelMetadata object to a ModelResponse."""
    return ModelResponse(
        id=model.id,
        name=model.name,
        provider=model.provider.value,
        version=model.version,
        status=model.status.value,
        capabilities=[cap.value for cap in model.capabilities],
        created_at=model.created_at,
        updated_at=model.updated_at
    )

# API endpoints
@router.get("/models", response_model=ModelListResponse)
async def list_models(
    provider: Optional[str] = None,
    model_manager: ModelManager = Depends(lambda: get_settings().model_manager)
) -> ModelListResponse:
    """List all available models, optionally filtered by provider.
    
    Args:
        provider: Optional provider name to filter by
        model_manager: Model manager instance
        
    Returns:
        List of available models
    """
    try:
        models = await model_manager.list_models(provider)
        return ModelListResponse(
            models=[model_to_response(model) for model in models]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )

@router.get("/models/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: str,
    model_manager: ModelManager = Depends(lambda: get_settings().model_manager)
) -> ModelResponse:
    """Get details about a specific model.
    
    Args:
        model_id: ID of the model to get
        model_manager: Model manager instance
        
    Returns:
        Model details
    """
    try:
        model = await model_manager.get_model(model_id)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model not found: {model_id}"
            )
        return model_to_response(model)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model: {str(e)}"
        )

@router.post("/models/load", response_model=ModelResponse)
async def load_model(
    request: LoadModelRequest,
    model_manager: ModelManager = Depends(lambda: get_settings().model_manager)
) -> ModelResponse:
    """Load a model into memory.
    
    Args:
        request: Load model request
        model_manager: Model manager instance
        
    Returns:
        Updated model details
    """
    try:
        model = await model_manager.load_model(
            request.model_id,
            **request.params
        )
        return model_to_response(model)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}"
        )

@router.post("/models/{model_id}/unload", response_model=ModelResponse)
async def unload_model(
    model_id: str,
    model_manager: ModelManager = Depends(lambda: get_settings().model_manager)
) -> ModelResponse:
    """Unload a model from memory.
    
    Args:
        model_id: ID of the model to unload
        model_manager: Model manager instance
        
    Returns:
        Updated model details
    """
    try:
        # First get the model to return its details after unloading
        model = await model_manager.get_model(model_id)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model not found: {model_id}"
            )
        
        # Unload the model
        success = await model_manager.unload_model(model_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to unload model: {model_id}"
            )
        
        # Return the model with updated status
        model.status = ModelStatus.UNLOADED
        return model_to_response(model)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unload model: {str(e)}"
        )

@router.post("/generate", response_model=dict)
async def generate_text(
    request: GenerateTextRequest,
    model_manager: ModelManager = Depends(lambda: get_settings().model_manager)
) -> dict:
    """Generate text using the specified model.
    
    Args:
        request: Generate text request
        model_manager: Model manager instance
        
    Returns:
        Generated text
    """
    try:
        # If no model_id is provided, use the first available model
        if not request.model_id:
            models = await model_manager.list_models()
            if not models:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No models available"
                )
            model_id = models[0].id
        else:
            model_id = request.model_id
        
        # Generate text
        text = await model_manager.generate_text(
            model_id=model_id,
            prompt=request.prompt,
            **request.params
        )
        
        return {
            "model_id": model_id,
            "text": text,
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate text: {str(e)}"
        )

@router.post("/chat", response_model=dict)
async def chat(
    request: ChatRequest,
    model_manager: ModelManager = Depends(lambda: get_settings().model_manager)
) -> dict:
    """Generate a chat completion using the specified model.
    
    Args:
        request: Chat request
        model_manager: Model manager instance
        
    Returns:
        Chat response
    """
    try:
        # If no model_id is provided, use the first available model that supports chat
        if not request.model_id:
            models = await model_manager.list_models()
            chat_models = [m for m in models if 'chat' in [c.value for c in m.capabilities]]
            
            if not chat_models:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No chat models available"
                )
            model_id = chat_models[0].id
        else:
            model_id = request.model_id
        
        # Convert messages to the format expected by the model manager
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Generate chat completion
        response = await model_manager.chat(
            model_id=model_id,
            messages=messages,
            **request.params
        )
        
        return {
            "model_id": model_id,
            "message": {
                "role": "assistant",
                "content": response
            },
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate chat completion: {str(e)}"
        )
