"""Ollama provider implementation for LLM MCP Server."""
import json
from typing import Any, Dict, List, Optional
import httpx
from datetime import datetime

from ...models.base import (
    BaseProvider,
    ModelMetadata,
    ModelProvider,
    ModelStatus,
    ModelCapability,
)

class OllamaProvider(BaseProvider):
    """Provider for Ollama models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Ollama provider.
        
        Args:
            config: Configuration dictionary with 'base_url' key
        """
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Any:
        """Make an HTTP request to the Ollama API."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = await self.client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            error_msg = f"Ollama API error: {e.response.text}"
            if e.response.status_code == 404:
                return None
            raise Exception(error_msg) from e
        except Exception as e:
            raise Exception(f"Failed to connect to Ollama API: {str(e)}") from e
    
    async def list_models(self) -> List[ModelMetadata]:
        """List all available Ollama models."""
        try:
            response = await self._request("GET", "/api/tags")
            if not response or "models" not in response:
                return []
            
            models = []
            for model_data in response["models"]:
                models.append(self._parse_model_metadata(model_data))
            
            return models
        except Exception as e:
            # If we can't connect to Ollama, return an empty list
            print(f"Error listing Ollama models: {str(e)}")
            return []
    
    async def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get details about a specific Ollama model."""
        try:
            # First check if the model exists by listing all models
            models = await self.list_models()
            for model in models:
                if model.id == model_id or model.name == model_id:
                    return model
            return None
        except Exception as e:
            print(f"Error getting Ollama model {model_id}: {str(e)}")
            return None
    
    async def load_model(self, model_id: str, **kwargs) -> ModelMetadata:
        """Load an Ollama model."""
        # In Ollama, models are loaded on first use, so we just need to check if it exists
        model = await self.get_model(model_id)
        if not model:
            # Try to pull the model if it doesn't exist
            return await self.pull_model(model_id, **kwargs)
        return model
    
    async def pull_model(self, model_id: str, **kwargs) -> ModelMetadata:
        """Pull a model from the Ollama library."""
        try:
            # Start the pull operation
            async with self.client.stream(
                "POST",
                "/api/pull",
                json={"name": model_id, "stream": True},
                timeout=300.0,  # 5 minutes timeout for model download
            ) as response:
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            # You can add progress tracking here if needed
                            if data.get("status") == "success":
                                break
                        except json.JSONDecodeError:
                            continue
            
            # After successful pull, get the model details
            return await self.get_model(model_id)
        except Exception as e:
            raise Exception(f"Failed to pull model {model_id}: {str(e)}") from e
    
    async def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory."""
        # Ollama doesn't have a direct unload endpoint
        # We'll just return True as the model will be unloaded when not in use
        return True
    
    async def generate_text(
        self,
        model_id: str,
        prompt: str,
        **kwargs
    ) -> str:
        """Generate text using the specified Ollama model."""
        try:
            data = {
                "model": model_id,
                "prompt": prompt,
                "stream": False,
                **kwargs
            }
            
            response = await self._request(
                "POST",
                "/api/generate",
                json=data,
                timeout=300.0
            )
            
            return response.get("response", "")
        except Exception as e:
            raise Exception(f"Failed to generate text with model {model_id}: {str(e)}") from e
    
    async def chat(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Generate a chat completion using the specified Ollama model."""
        try:
            data = {
                "model": model_id,
                "messages": messages,
                "stream": False,
                **kwargs
            }
            
            response = await self._request(
                "POST",
                "/api/chat",
                json=data,
                timeout=300.0
            )
            
            # Extract the assistant's message from the response
            if response and "message" in response and "content" in response["message"]:
                return response["message"]["content"]
            return ""
        except Exception as e:
            raise Exception(f"Failed to chat with model {model_id}: {str(e)}") from e
    
    async def generate_embeddings(
        self,
        model_id: str,
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings for the given texts using Ollama."""
        try:
            # Ollama doesn't support batch embeddings, so we'll process one by one
            embeddings = []
            for text in texts:
                data = {
                    "model": model_id,
                    "prompt": text,
                    **kwargs
                }
                
                response = await self._request(
                    "POST",
                    "/api/embeddings",
                    json=data,
                    timeout=60.0
                )
                
                if response and "embedding" in response:
                    embeddings.append(response["embedding"])
                else:
                    embeddings.append([])
            
            return embeddings
        except Exception as e:
            raise Exception(f"Failed to generate embeddings with model {model_id}: {str(e)}") from e
    
    def _parse_model_metadata(self, model_data: Dict[str, Any]) -> ModelMetadata:
        """Parse Ollama model data into a ModelMetadata object."""
        # Extract model name and tag
        model_name = model_data.get("name", "")
        model_id = model_data.get("model", model_name)
        
        # Parse model details
        details = model_data.get("details", {})
        
        # Determine model capabilities based on model family
        capabilities = [ModelCapability.TEXT_GENERATION, ModelCapability.CHAT]
        family = details.get("family", "").lower()
        
        if "llava" in family or "clip" in family:
            capabilities.append(ModelCapability.VISION)
        
        if "nomic" in family or "embed" in model_name.lower():
            capabilities.append(ModelCapability.EMBEDDINGS)
        
        return ModelMetadata(
            id=model_id,
            name=model_name,
            provider=ModelProvider.OLLAMA,
            version=model_data.get("modified_at", ""),
            status=ModelStatus.LOADED if "size" in model_data else ModelStatus.UNLOADED,
            capabilities=capabilities,
            parameters={
                "size": model_data.get("size"),
                "digest": model_data.get("digest"),
                "family": details.get("family"),
                "parameter_size": details.get("parameter_size"),
                "quantization_level": details.get("quantization_level"),
            },
            created_at=model_data.get("modified_at"),
            updated_at=model_data.get("modified_at"),
        )
