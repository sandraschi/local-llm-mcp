"""Ollama provider implementation for LLM MCP.

This module provides integration with Ollama's local LLM service.
"""
from typing import Dict, Any, List, AsyncGenerator
import httpx
from ..base import BaseProvider

class OllamaProvider(BaseProvider):
    """Provider for interacting with Ollama's local LLM service."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Ollama provider.
        
        Args:
            config: Configuration dictionary containing:
                - base_url: Base URL for the Ollama API (default: http://localhost:11434)
                - timeout: Request timeout in seconds (default: 60)
        """
        super().__init__(config)
        self.base_url = self.config.get("base_url", "http://localhost:11434")
        self.timeout = self.config.get("timeout", 60)
        self.client = httpx.AsyncClient(timeout=self.timeout)
    
    @property
    def name(self) -> str:
        """Return the name of the provider."""
        return "ollama"
    
    @property
    def supports_streaming(self) -> bool:
        """Return whether the provider supports streaming responses."""
        return True
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all available models from the Ollama server.
        
        Returns:
            List of model information dictionaries.
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            
            return [
                {
                    "id": model["name"],
                    "name": model["name"],
                    "description": f"Ollama model: {model['name']}",
                    "size": model.get("size"),
                    "modified_at": model.get("modified_at"),
                    "capabilities": ["generate", "stream"],
                }
                for model in data.get("models", [])
            ]
        except httpx.HTTPStatusError as e:
            raise Exception(f"Failed to list models: {e.response.text}") from e
        except Exception as e:
            raise Exception(f"Error listing models: {str(e)}") from e
    
    async def generate(
        self,
        prompt: str,
        model: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate a response from the Ollama model.
        
        Args:
            prompt: The input prompt
            model: The model to use for generation
            **kwargs: Additional generation parameters:
                - system: System prompt
                - template: Template to use
                - context: Context from previous messages
                - format: Format to return response in (e.g., json)
                - options: Additional model options
                
        Yields:
            Chunks of the generated response as strings
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            **{
                k: v for k, v in kwargs.items()
                if k in ["system", "template", "context", "format", "options"]
            }
        }
        
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = httpx._utils.json.loads(line)
                            if "response" in data:
                                yield data["response"]
                            if data.get("done", False):
                                break
                        except Exception as e:
                            raise Exception(f"Error parsing response: {str(e)}") from e
        except Exception as e:
            raise Exception(f"Error during generation: {str(e)}") from e
    
    async def pull_model(self, model_name: str) -> Dict[str, Any]:
        """Download a model from the Ollama library.
        
        Args:
            model_name: Name of the model to download
            
        Returns:
            Status information about the download operation
        """
        url = f"{self.base_url}/api/pull"
        payload = {"name": model_name}
        
        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            
            # If we get here, the pull was successful
            return {
                "status": "success",
                "model": model_name,
                "message": f"Successfully pulled {model_name}"
            }
        except httpx.HTTPStatusError as e:
            raise Exception(f"Failed to pull model: {e.response.text}") from e
        except Exception as e:
            raise Exception(f"Error pulling model: {str(e)}") from e
    
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model to get info for
            
        Returns:
            Detailed model information
        """
        # First get the list of models
        models = await self.list_models()
        
        # Find the requested model
        for model in models:
            if model["name"] == model_name:
                return model
        
        # If model not found, try to get its details directly
        try:
            url = f"{self.base_url}/api/show"
            response = await self.client.post(url, json={"name": model_name})
            response.raise_for_status()
            
            data = response.json()
            return {
                "id": data.get("name"),
                "name": data.get("name"),
                "description": f"Ollama model: {data.get('name')}",
                "parameters": data.get("parameters"),
                "template": data.get("template"),
                "system": data.get("system"),
                "license": data.get("license"),
                "capabilities": ["generate", "stream"],
            }
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Model '{model_name}' not found") from e
            raise Exception(f"Failed to get model info: {e.response.text}") from e
        except Exception as e:
            raise Exception(f"Error getting model info: {str(e)}") from e
    
    async def close(self):
        """Clean up resources."""
        await self.client.aclose()
    
    def __del__(self):
        """Ensure resources are cleaned up when the object is destroyed."""
        try:
            import asyncio
            asyncio.create_task(self.close())
        except:
            pass
