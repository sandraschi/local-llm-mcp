"""LM Studio provider implementation for LLM MCP Server."""
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

from ...models.base import (
    BaseProvider,
    ModelMetadata,
    ModelProvider,
    ModelStatus,
    ModelCapability,
)

logger = logging.getLogger(__name__)

class LMStudioProvider(BaseProvider):
    """Provider for LM Studio models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the LM Studio provider.
        
        Args:
            config: Configuration dictionary with 'base_url' key
        """
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:1234")
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Any:
        """Make an HTTP request to the LM Studio API."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = await self.client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            error_msg = f"LM Studio API error: {e.response.text}"
            if e.response.status_code == 404:
                return None
            logger.error(error_msg)
            raise Exception(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to connect to LM Studio API: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
    
    async def list_models(self) -> List[ModelMetadata]:
        """List all available LM Studio models."""
        try:
            # LM Studio uses the OpenAI-compatible API
            response = await self._request("GET", "/v1/models")
            if not response or "data" not in response:
                return []
            
            models = []
            for model_data in response["data"]:
                models.append(self._parse_model_metadata(model_data))
            
            return models
        except Exception as e:
            logger.error(f"Error listing LM Studio models: {str(e)}")
            return []
    
    async def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get details about a specific LM Studio model."""
        try:
            # First check if the model exists by listing all models
            models = await self.list_models()
            for model in models:
                if model.id == model_id or model.name == model_id:
                    return model
            return None
        except Exception as e:
            logger.error(f"Error getting LM Studio model {model_id}: {str(e)}")
            return None
    
    async def load_model(self, model_id: str, **kwargs) -> ModelMetadata:
        """Load an LM Studio model.
        
        In LM Studio, models are loaded on first use, so we just need to check
        if the model exists.
        """
        model = await self.get_model(model_id)
        if not model:
            raise ValueError(f"Model not found: {model_id}")
        
        # Update model status to loaded
        model.status = ModelStatus.LOADED
        return model
    
    async def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory.
        
        Note: LM Studio doesn't support unloading individual models through the API.
        This is a no-op that always returns True.
        """
        return True
    
    async def generate_text(
        self,
        model_id: str,
        prompt: str,
        **kwargs
    ) -> str:
        """Generate text using the specified LM Studio model."""
        try:
            data = {
                "model": model_id,
                "prompt": prompt,
                "stream": False,
                **kwargs
            }
            
            response = await self._request(
                "POST",
                "/v1/completions",
                json=data,
                timeout=300.0
            )
            
            return response.get("choices", [{}])[0].get("text", "")
        except Exception as e:
            error_msg = f"Failed to generate text with model {model_id}: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
    
    async def chat(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Generate a chat completion using the specified LM Studio model."""
        try:
            data = {
                "model": model_id,
                "messages": messages,
                "stream": False,
                **kwargs
            }
            
            response = await self._request(
                "POST",
                "/v1/chat/completions",
                json=data,
                timeout=300.0
            )
            
            # Extract the assistant's message from the response
            if response and "choices" in response and len(response["choices"]) > 0:
                return response["choices"][0]["message"].get("content", "")
            return ""
        except Exception as e:
            error_msg = f"Failed to chat with model {model_id}: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
    
    async def generate_embeddings(
        self,
        model_id: str,
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings for the given texts using LM Studio."""
        try:
            # LM Studio uses the OpenAI-compatible embeddings API
            embeddings = []
            
            for text in texts:
                data = {
                    "model": model_id,
                    "input": text,
                    **kwargs
                }
                
                response = await self._request(
                    "POST",
                    "/v1/embeddings",
                    json=data,
                    timeout=60.0
                )
                
                if response and "data" in response and len(response["data"]) > 0:
                    embeddings.append(response["data"][0]["embedding"])
                else:
                    embeddings.append([])
            
            return embeddings
        except Exception as e:
            error_msg = f"Failed to generate embeddings with model {model_id}: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
    
    def _parse_model_metadata(self, model_data: Dict[str, Any]) -> ModelMetadata:
        """Parse LM Studio model data into a ModelMetadata object."""
        model_id = model_data.get("id", "")
        model_name = model_data.get("name", model_id)
        
        # Default capabilities
        capabilities = [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.EMBEDDINGS
        ]
        
        # Check if model supports vision (this is a heuristic)
        if "vision" in model_name.lower() or "clip" in model_name.lower():
            capabilities.append(ModelCapability.VISION)
        
        return ModelMetadata(
            id=model_id,
            name=model_name,
            provider=ModelProvider.LMSTUDIO,
            version=model_data.get("version", "1.0"),
            status=ModelStatus.LOADED,  # LM Studio models are always loaded
            capabilities=capabilities,
            parameters={
                "owned_by": model_data.get("owned_by", "lmstudio"),
                "permissions": model_data.get("permission", []),
                "context_length": model_data.get("context_length", 2048),
            },
            created_at=model_data.get("created", ""),
            updated_at=model_data.get("updated", "")
        )
    
    async def close(self):
        """Clean up resources."""
        await self.client.aclose()
