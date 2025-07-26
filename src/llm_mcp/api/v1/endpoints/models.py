"""API endpoints for model management."""
from typing import List, Optional, Dict, Any, AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import StreamingResponse
import json

from ....services.model_service import model_service
from ....config import get_settings
from ..models import (
    ProviderInfo,
    ModelInfo,
    GenerateRequest,
    GenerateResponse,
    ModelOperationResponse,
    ModelStatus
)

# Create router
router = APIRouter()

# API endpoints
@router.get("/providers", response_model=List[ProviderInfo])
async def list_providers() -> List[ProviderInfo]:
    """List all available providers.
    
    Returns:
        List of available providers with their capabilities
    """
    try:
        settings = get_settings()
        providers = []
        
        for provider_name, config in settings.providers.items():
            if not config.get("enabled", True):
                continue
                
            providers.append(ProviderInfo(
                name=provider_name,
                description=f"{provider_name.capitalize()} provider",
                capabilities=["generate", "stream"]
            ))
            
        return providers
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list providers: {str(e)}"
        )

@router.get("/models", response_model=List[ModelInfo])
async def list_models(provider: Optional[str] = None) -> List[ModelInfo]:
    """List all available models, optionally filtered by provider.
    
    Args:
        provider: Optional provider name to filter by
        
    Returns:
        List of available models with their details
    """
    try:
        return await model_service.list_models(provider)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )

@router.get("/models/{model_name}", response_model=ModelInfo)
async def get_model(
    model_name: str,
    provider: Optional[str] = None
) -> ModelInfo:
    """Get detailed information about a specific model.
    
    Args:
        model_name: Name of the model to get info for
        provider: Optional provider name if known
        
    Returns:
        Detailed model information
    """
    try:
        return await model_service.get_model_info(model_name, provider)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )

@router.post("/models/pull", response_model=ModelOperationResponse)
async def pull_model(
    model_name: str,
    provider: Optional[str] = None
) -> ModelOperationResponse:
    """Download a model if it's not already available locally.
    
    Args:
        model_name: Name of the model to download
        provider: Optional provider name if known
        
    Returns:
        Status of the download operation
    """
    try:
        if not provider:
            provider = get_settings().default_provider
            
        provider_instance = model_service.providers.get(provider.lower())
        if not provider_instance:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Provider not available: {provider}"
            )
            
        if not hasattr(provider_instance, 'pull_model'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Provider {provider} does not support pulling models"
            )
            
        result = await provider_instance.pull_model(model_name)
        return ModelOperationResponse(
            model=model_name,
            provider=provider,
            status=ModelStatus.READY,
            message=result.get("message", "Model downloaded successfully"),
            details=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to pull model: {str(e)}"
        )

@router.post("/generate", response_model=GenerateResponse)
async def generate_text(
    request: GenerateRequest,
    raw_request: Request
) -> GenerateResponse:
    """Generate text using the specified model.
    
    Args:
        request: Generate text request
        
    Returns:
        Generated text response
    """
    # If streaming is requested, return a streaming response
    if request.stream:
        return StreamingResponse(
            generate_stream(request, raw_request),
            media_type="text/event-stream"
        )
    
    # Otherwise, generate the full response at once
    try:
        full_response = ""
        async for chunk in model_service.generate(
            prompt=request.prompt,
            model=request.model,
            provider=request.provider,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            stop=request.stop
        ):
            full_response += chunk
            
        return GenerateResponse(
            text=full_response,
            model=request.model,
            provider=request.provider or get_settings().default_provider,
            usage={
                "prompt_tokens": len(request.prompt.split()),  # Approximate
                "completion_tokens": len(full_response.split()),  # Approximate
                "total_tokens": len(request.prompt.split()) + len(full_response.split())
            }
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate text: {str(e)}"
        )

async def generate_stream(
    request: GenerateRequest,
    raw_request: Request
) -> AsyncGenerator[bytes, None]:
    """Generate text in a streaming fashion.
    
    Args:
        request: Generate text request
        raw_request: The raw request object for checking if the client disconnected
        
    Yields:
        Chunks of the generated text as server-sent events
    """
    try:
        buffer = ""
        
        async for chunk in model_service.generate(
            prompt=request.prompt,
            model=request.model,
            provider=request.provider,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            stop=request.stop
        ):
            # Check if client disconnected
            if await raw_request.is_disconnected():
                break
                
            # Add to buffer and yield if we have a complete line
            buffer += chunk
            if "\n" in buffer:
                lines = buffer.split("\n")
                for line in lines[:-1]:
                    yield f"data: {json.dumps({'text': line})}\n\n".encode()
                buffer = lines[-1]
            else:
                yield f"data: {json.dumps({'text': chunk})}\n\n".encode()
                
        # Yield any remaining content in the buffer
        if buffer:
            yield f"data: {json.dumps({'text': buffer})}\n\n".encode()
            
        # Send the [DONE] event
        yield "data: [DONE]\n\n".encode()
        
    except Exception as e:
        error_msg = f"Error during streaming: {str(e)}"
        yield f"data: {json.dumps({'error': error_msg})}\n\n".encode()
