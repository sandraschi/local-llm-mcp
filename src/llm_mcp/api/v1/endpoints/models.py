"""API endpoints for model management."""
from typing import List, Optional, Dict, Any, AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, status, Request, Query
from fastapi.responses import StreamingResponse, JSONResponse
import json
import logging

from ....services.model_service import model_service
from ....config import get_settings
from ....providers.base import BaseProvider
from ..models import (
    ProviderInfo,
    ModelInfo,
    GenerateRequest,
    GenerateResponse,
    ModelOperationResponse,
    ModelStatus
)

logger = logging.getLogger(__name__)

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
        
        # Add vLLM provider if available
        try:
            from ....providers.vllm_v1.provider import VLLMv1Provider
            providers.append(ProviderInfo(
                name="vllm",
                description="vLLM provider for high-performance LLM inference",
                capabilities=["generate", "stream", "batch", "embeddings"],
                parameters={
                    "model": {"type": "string", "required": True, "description": "Model name or path"},
                    "tensor_parallel_size": {"type": "integer", "required": False, "default": 1, "description": "Number of GPUs to use"},
                    "gpu_memory_utilization": {"type": "float", "required": False, "default": 0.9, "description": "Fraction of GPU memory to use"},
                    "max_seq_len": {"type": "integer", "required": False, "default": 2048, "description": "Maximum sequence length"},
                    "quantization": {"type": "string", "required": False, "enum": [None, "awq", "gptq", "squeezellm"], "description": "Quantization method"},
                }
            ))
        except ImportError:
            logger.warning("vLLM provider not available")
        
        # Add other configured providers
        for provider_name, config in settings.providers.items():
            if not config.get("enabled", True) or provider_name in ("vllm", "vllm_v1"):
                continue
                
            providers.append(ProviderInfo(
                name=provider_name,
                description=f"{provider_name.capitalize()} provider",
                capabilities=["generate", "stream"]
            ))
            
        return providers
        
    except Exception as e:
        logger.error(f"Failed to list providers: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list providers: {str(e)}"
        )

@router.get("/models", response_model=List[ModelInfo])
async def list_models(
    provider: Optional[str] = None,
    include_vllm: bool = Query(True, description="Include vLLM models in the results")
) -> List[ModelInfo]:
    """List all available models, optionally filtered by provider.
    
    Args:
        provider: Optional provider name to filter models by
        include_vllm: Whether to include vLLM models in the results
        
    Returns:
        List of available models with their details
    """
    try:
        models = []
        
        # Handle vLLM models separately if requested
        if (not provider or provider.lower() in ('vllm', 'vllm_v1')) and include_vllm:
            try:
                from ....providers.vllm_v1.provider import VLLMv1Provider
                
                # Get vLLM provider instance or create a temporary one
                vllm_provider = None
                if 'vllm' in model_service.providers:
                    vllm_provider = model_service.providers['vllm']
                elif 'vllm_v1' in model_service.providers:
                    vllm_provider = model_service.providers['vllm_v1']
                else:
                    # Create a temporary vLLM provider to list models
                    vllm_provider = VLLMv1Provider({})
                    
                # Get vLLM models
                vllm_models = await vllm_provider.list_models()
                models.extend([
                    ModelInfo(
                        id=model.get('id'),
                        name=model.get('name', ''),
                        description=model.get('description', ''),
                        provider='vllm',
                        capabilities=model.get('capabilities', ['generate', 'stream']),
                        parameters=model.get('parameters', {})
                    )
                    for model in vllm_models
                ])
            except ImportError:
                logger.warning("vLLM provider not available")
            except Exception as e:
                logger.error(f"Error listing vLLM models: {str(e)}", exc_info=True)
        
        # Get models from other providers
        if provider and provider.lower() not in ('vllm', 'vllm_v1'):
            # Specific provider requested (not vLLM)
            try:
                provider_models = await model_service.list_models(provider)
                models.extend([
                    ModelInfo(
                        id=model.get('id'),
                        name=model.get('name', ''),
                        description=model.get('description', ''),
                        provider=provider,
                        capabilities=model.get('capabilities', ['generate', 'stream']),
                        parameters=model.get('parameters', {})
                    )
                    for model in provider_models
                ])
            except Exception as e:
                logger.error(f"Error listing models for provider {provider}: {str(e)}")
        elif not provider:
            # No specific provider requested, get all providers
            for provider_name, provider_instance in model_service.providers.items():
                try:
                    if provider_name.lower() in ('vllm', 'vllm_v1'):
                        continue  # Already handled above
                        
                    provider_models = await model_service.list_models(provider_name)
                    models.extend([
                        ModelInfo(
                            id=model.get('id'),
                            name=model.get('name', ''),
                            description=model.get('description', ''),
                            provider=provider_name,
                            capabilities=model.get('capabilities', ['generate', 'stream']),
                            parameters=model.get('parameters', {})
                        )
                        for model in provider_models
                    ])
                except Exception as e:
                    logger.error(f"Error listing models for provider {provider_name}: {str(e)}")
        
        return models
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}", exc_info=True)
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
        logger.error(f"Failed to get model info for {model_name}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )

@router.post("/models/pull", response_model=ModelOperationResponse)
async def pull_model(
    model_name: str,
    provider: Optional[str] = None,
    force: bool = Query(False, description="Force download even if model exists"),
    **kwargs: Any
) -> ModelOperationResponse:
    """Download a model if it's not already available locally.
    
    Args:
        model_name: Name of the model to download
        provider: Optional provider name if known
        force: Force download even if model exists
        **kwargs: Additional provider-specific parameters
        
    Returns:
        Status of the download operation
    """
    try:
        # Special handling for vLLM provider
        if not provider or provider.lower() in ('vllm', 'vllm_v1'):
            try:
                from ....providers.vllm_v1.provider import VLLMv1Provider
                
                # Get or create vLLM provider
                vllm_provider = None
                if 'vllm' in model_service.providers:
                    vllm_provider = model_service.providers['vllm']
                elif 'vllm_v1' in model_service.providers:
                    vllm_provider = model_service.providers['vllm_v1']
                else:
                    # Create a temporary vLLM provider with provided kwargs
                    vllm_config = {k: v for k, v in kwargs.items() if v is not None}
                    vllm_provider = VLLMv1Provider(vllm_config)
                
                # Check if model exists in vLLM
                vllm_models = await vllm_provider.list_models()
                model_exists = any(m.get('id') == model_name for m in vllm_models)
                
                if model_exists and not force:
                    return ModelOperationResponse(
                        success=True,
                        message=f"Model {model_name} already exists in vLLM",
                        details={"status": "already_exists"}
                    )
                
                # Pull the model
                result = await vllm_provider.pull_model(model_name, **kwargs)
                
                return ModelOperationResponse(
                    success=True,
                    message=f"Successfully pulled model {model_name} using vLLM",
                    details=result
                )
                
            except ImportError:
                if provider:  # Only raise if vLLM was explicitly requested
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="vLLM provider is not available"
                    )
            except Exception as e:
                logger.error(f"Error pulling vLLM model {model_name}: {str(e)}", exc_info=True)
                if provider:  # Only raise if vLLM was explicitly requested
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to pull vLLM model: {str(e)}"
                    )
        
        # If we get here, either vLLM is not the provider or the pull failed
        try:
            # Get the provider for this model
            provider_instance, provider_name = await model_service.get_provider_for_model(model_name, provider)
            if not provider_instance:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model not found: {model_name}"
                )
            
            # Pull the model
            result = await provider_instance.pull_model(model_name, **kwargs)
            
            return ModelOperationResponse(
                success=True,
                message=f"Successfully pulled model {model_name} from {provider_name}",
                details=result
            )
            
        except HTTPException:
            raise
            
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
