"""LM Studio provider implementation for LLM MCP Server."""
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, AsyncGenerator

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
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models from LM Studio.
        
        Returns:
            List of model information dictionaries
        """
        try:
            # LM Studio doesn't have a dedicated models endpoint,
            # so we return the currently loaded model if any
            models = []
            model_info = await self.get_model_info()
            if model_info:
                models.append({
                    "id": "lmstudio-default",
                    "name": model_info.get("id", "lmstudio-model"),
                    "description": f"LM Studio model: {model_info.get('id', 'default')}",
                    "capabilities": ["text-generation", "chat"]
                })
            return models
        except Exception as e:
            logger.error(f"Failed to list LM Studio models: {str(e)}")
            return []
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate text using the LM Studio API.
        
        Args:
            prompt: The input prompt
            model: Model to use (ignored for LM Studio)
            **kwargs: Additional generation parameters
            
        Yields:
            Chunks of the generated text
        """
        try:
            # LM Studio uses the OpenAI-compatible API
            endpoint = "/v1/chat/completions"
            payload = {
                "model": "local-model",  # LM Studio uses this as a placeholder
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                **{k: v for k, v in kwargs.items() if v is not None}
            }
            
            async with self.client.stream(
                "POST",
                endpoint,
                json=payload,
                timeout=60.0
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        chunk = line[6:].strip()
                        if chunk == "[DONE]":
                            break
                            
                        try:
                            data = json.loads(chunk)
                            if "choices" in data and len(data["choices"]) > 0:
                                content = data["choices"][0].get("delta", {}).get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse LM Studio response: {chunk}")
                            
        except Exception as e:
            logger.error(f"Error in LM Studio generation: {str(e)}")
            raise
    
    async def pull_model(self, model_name: str) -> Dict[str, Any]:
        """Pull a model from LM Studio.
        
        Note: LM Studio doesn't support pulling models programmatically.
        This method is a placeholder for API compatibility.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            Dictionary with pull status
        """
        return {
            "status": "not_supported",
            "message": "LM Studio doesn't support pulling models programmatically",
            "model": model_name
        }
    
    async def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about the currently loaded model.
        
        Args:
            model_name: Model name (ignored for LM Studio)
            
        Returns:
            Dictionary with model information
        """
        try:
            # LM Studio provides model info in the completions endpoint
            endpoint = "/v1/models"
            response = await self._request("GET", endpoint)
            
            if response and "data" in response and len(response["data"]) > 0:
                return response["data"][0]
                
            return {
                "id": "lmstudio-model",
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "lmstudio"
            }
            
        except Exception as e:
            logger.error(f"Failed to get LM Studio model info: {str(e)}")
            return {
                "id": "lmstudio-model",
                "error": str(e)
            }
    
    async def close(self):
        """Clean up resources."""
        if hasattr(self, 'client'):
            await self.client.aclose()
