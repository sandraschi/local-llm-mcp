"""Service for managing LLM models and providers."""
import logging
from typing import Dict, Any, Optional, AsyncGenerator, List

from ..providers import ProviderFactory
from ..config import get_settings, Settings

logger = logging.getLogger(__name__)

class ModelService:
    """Service for managing LLM models and providers."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the model service.
        
        Args:
            settings: Application settings. If not provided, will use default settings.
        """
        self.settings = settings or get_settings()
        self.providers: Dict[str, Any] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the model service and all providers."""
        if self._initialized:
            return
            
        logger.info("Initializing ModelService...")
        
        # Initialize all configured providers
        for provider_name, config in self.settings.providers.items():
            if not config.get("enabled", True):
                logger.info(f"Skipping disabled provider: {provider_name}")
                continue
                
            try:
                # Special handling for vLLM provider
                if provider_name.lower() in ('vllm', 'vllm_v1'):
                    provider = await self._initialize_vllm_provider(provider_name, config)
                else:
                    provider = await ProviderFactory.create_and_verify_provider(
                        provider_name,
                        config
                    )
                
                if provider:
                    self.providers[provider_name] = provider
                    logger.info(f"Initialized provider: {provider_name}")
                else:
                    logger.warning(f"Failed to initialize provider: {provider_name}")
                    
            except Exception as e:
                logger.error(f"Error initializing provider {provider_name}: {str(e)}", exc_info=True)
    
    async def _initialize_vllm_provider(self, provider_name: str, config: Dict[str, Any]) -> Any:
        """Special initialization for vLLM provider.
        
        Args:
            provider_name: Name of the vLLM provider ('vllm' or 'vllm_v1')
            config: Provider configuration
            
        Returns:
            Initialized vLLM provider instance or None if initialization fails
        """
        try:
            # Import here to avoid circular imports
            from ..providers.vllm_v1.provider import VLLMv1Provider
            
            # Create provider instance
            provider = VLLMv1Provider(config)
            
            # Initialize the provider
            await provider.initialize()
            
            return provider
            
        except ImportError as e:
            logger.warning(f"vLLM provider not available: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize vLLM provider: {str(e)}", exc_info=True)
            return None
    
    async def get_provider_for_model(self, model_name: str, provider_name: str = None) -> Any:
        """Get the appropriate provider for the given model.
        
        Args:
            model_name: Name of the model
            provider_name: Optional provider name if known
            
        Returns:
            A tuple of (provider, provider_name) or (None, None) if not found
            
        Raises:
            HTTPException: If the provider is not available or fails to initialize
        """
        # If provider is specified, use it
        if provider_name:
            provider = self.providers.get(provider_name.lower())
            if not provider:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Provider not found: {provider_name}"
                )
            return provider, provider_name
            
        # Otherwise, try to find a provider that supports this model
        for name, provider in self.providers.items():
            try:
                models = await provider.list_models()
                if any(m.get('id') == model_name for m in models):
                    return provider, name
            except Exception as e:
                logger.warning(f"Error listing models for provider {name}: {str(e)}")
                continue
                
        return None, None
        
        self._initialized = True
        logger.info("ModelService initialization complete")
    
    async def list_models(self, provider_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all available models from all or a specific provider.
        
        Args:
            provider_name: Optional name of the provider to list models from.
                          If None, lists models from all providers.
                          
        Returns:
            List of model information dictionaries.
        """
        if not self._initialized:
            await self.initialize()
            
        models = []
        
        # If a specific provider is requested
        if provider_name:
            provider = self.providers.get(provider_name.lower())
            if not provider:
                raise ValueError(f"Provider not found or not initialized: {provider_name}")
                
            provider_models = await provider.list_models()
            for model in provider_models:
                model["provider"] = provider_name
                models.append(model)
        else:
            # List models from all providers
            for provider_name, provider in self.providers.items():
                try:
                    provider_models = await provider.list_models()
                    for model in provider_models:
                        model["provider"] = provider_name
                        models.append(model)
                except Exception as e:
                    logger.error(f"Error listing models from provider {provider_name}: {str(e)}", exc_info=True)
        
        return models
    
    async def generate(
        self,
        prompt: str,
        model: str,
        provider: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 100,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate text using the specified model.
        
        Args:
            prompt: The prompt to generate text from
            model: The model to use for generation
            provider: Optional provider name if known
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter (0-1)
            frequency_penalty: Penalty for token frequency (-2.0 to 2.0)
            presence_penalty: Penalty for new tokens (-2.0 to 2.0)
            stop: List of strings that stop generation when encountered
            **kwargs: Additional provider-specific parameters
                For vLLM, this can include:
                - top_k: Integer that controls the number of top tokens to consider
                - best_of: Integer that controls the number of completions to generate and return the best one
                - use_beam_search: Boolean to enable beam search
                - length_penalty: Float that penalizes longer sequences
                - early_stopping: Boolean or string for early stopping
                - stop_token_ids: List of token IDs to stop generation
                - ignore_eos: Boolean to ignore EOS token
                - logprobs: Integer specifying how many top token log probabilities to return
                - prompt_logprobs: Integer specifying how many top token log probabilities to return for the prompt
            
        Yields:
            Generated text chunks
            
        Raises:
            ValueError: If the model or provider is not found
            HTTPException: If there's an error during generation
        """
        try:
            # Special handling for vLLM provider
            if not provider or provider.lower() in ('vllm', 'vllm_v1'):
                try:
                    from ..providers.vllm_v1.provider import VLLMv1Provider
                    
                    # Get or create vLLM provider
                    vllm_provider = None
                    if 'vllm' in self.providers:
                        vllm_provider = self.providers['vllm']
                    elif 'vllm_v1' in self.providers:
                        vllm_provider = self.providers['vllm_v1']
                    else:
                        # Create a temporary vLLM provider with provided kwargs
                        vllm_config = {k: v for k, v in kwargs.items() 
                                     if k in ['model', 'tensor_parallel_size', 'gpu_memory_utilization', 
                                             'max_seq_len', 'quantization']}
                        vllm_provider = VLLMv1Provider(vllm_config)
                        await vllm_provider.initialize()
                    
                    # Prepare vLLM specific parameters
                    vllm_params = {
                        'temperature': temperature,
                        'top_p': top_p,
                        'max_tokens': max_tokens,
                        'frequency_penalty': frequency_penalty,
                        'presence_penalty': presence_penalty,
                        'stop': stop,
                    }
                    
                    # Add vLLM specific parameters from kwargs
                    vllm_specific_params = [
                        'top_k', 'best_of', 'use_beam_search', 'length_penalty',
                        'early_stopping', 'stop_token_ids', 'ignore_eos',
                        'logprobs', 'prompt_logprobs'
                    ]
                    for param in vllm_specific_params:
                        if param in kwargs:
                            vllm_params[param] = kwargs[param]
                    
                    # Generate using vLLM
                    async for chunk in vllm_provider.generate(
                        prompt=prompt,
                        model=model,
                        **vllm_params
                    ):
                        yield chunk
                    
                    return
                    
                except ImportError:
                    if provider:  # Only continue if vLLM was explicitly requested
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="vLLM provider is not available"
                        )
                except Exception as e:
                    logger.error(f"vLLM generation error: {str(e)}", exc_info=True)
                    if provider:  # Only raise if vLLM was explicitly requested
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"vLLM generation failed: {str(e)}"
                        )
            
            # If we get here, either vLLM is not the provider or the generation failed
            try:
                # Get the provider for this model
                provider_instance, _ = await self.get_provider_for_model(model, provider)
                if not provider_instance:
                    raise ValueError(f"Model not found: {model}")
                    
                # Generate the text using the standard provider interface
                async for chunk in provider_instance.generate(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                    **kwargs
                ):
                    yield chunk
                    
            except HTTPException:
                raise
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Generation error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate text: {str(e)}"
            )
                
        except Exception as e:
            logger.error(f"Error during generation with {provider_name}/{model}: {str(e)}", exc_info=True)
            raise
    
    async def get_model_info(self, model_name: str, provider_name: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model
            provider_name: Optional name of the provider. If None, searches all providers.
            
        Returns:
            Detailed model information
            
        Raises:
            ValueError: If the model is not found
        """
        if not self._initialized:
            await self.initialize()
        
        # If provider is specified, only check that provider
        if provider_name:
            provider = self.providers.get(provider_name.lower())
            if not provider:
                raise ValueError(f"Provider not found: {provider_name}")
                
            try:
                model_info = await provider.get_model_info(model_name)
                model_info["provider"] = provider_name
                return model_info
            except Exception as e:
                raise ValueError(f"Error getting model info: {str(e)}")
        
        # Otherwise, search all providers
        for provider_name, provider in self.providers.items():
            try:
                model_info = await provider.get_model_info(model_name)
                model_info["provider"] = provider_name
                return model_info
            except (ValueError, KeyError):
                continue
            except Exception as e:
                logger.warning(f"Error getting model info from {provider_name}: {str(e)}")
        
        raise ValueError(f"Model not found: {model_name}")
    
    async def close(self) -> None:
        """Clean up resources."""
        for provider in self.providers.values():
            if hasattr(provider, 'close') and callable(provider.close):
                await provider.close()
        
        self.providers.clear()
        self._initialized = False


# Create a singleton instance of the model service
model_service = ModelService()
