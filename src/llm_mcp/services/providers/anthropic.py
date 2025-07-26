"""Anthropic provider implementation for LLM MCP Server."""
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from ...models.base import (
    BaseProvider,
    ModelMetadata,
    ModelProvider,
    ModelStatus,
    ModelCapability,
)

logger = logging.getLogger(__name__)

class AnthropicMessage(BaseModel):
    role: str
    content: str

class AnthropicProvider(BaseProvider):
    """Provider for Anthropic's Claude models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Anthropic provider.
        
        Args:
            config: Configuration dictionary with 'api_key' and optional 'base_url'
        """
        super().__init__(config)
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
            
        self.base_url = config.get("base_url", "https://api.anthropic.com")
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            timeout=60.0
        )
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Any:
        """Make an HTTP request to the Anthropic API."""
        try:
            response = await self.client.request(method, endpoint, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            error_msg = f"Anthropic API error: {e.response.text}"
            logger.error(error_msg)
            if e.response.status_code == 401:
                raise ValueError("Invalid Anthropic API key") from e
            if e.response.status_code == 404:
                return None
            raise Exception(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to connect to Anthropic API: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
    
    async def list_models(self) -> List[ModelMetadata]:
        """List all available Anthropic models."""
        # Anthropic doesn't have a models endpoint, so we return known models
        known_models = [
            {
                "id": "claude-3-opus-20240229",
                "name": "Claude 3 Opus",
                "context_length": 200000,
                "capabilities": [
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT,
                    ModelCapability.VISION
                ]
            },
            {
                "id": "claude-3-sonnet-20240229",
                "name": "Claude 3 Sonnet",
                "context_length": 200000,
                "capabilities": [
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT,
                    ModelCapability.VISION
                ]
            },
            {
                "id": "claude-3-haiku-20240307",
                "name": "Claude 3 Haiku",
                "context_length": 200000,
                "capabilities": [
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT,
                    ModelCapability.VISION
                ]
            },
            {
                "id": "claude-2.1",
                "name": "Claude 2.1",
                "context_length": 100000,
                "capabilities": [
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT
                ]
            },
            {
                "id": "claude-2.0",
                "name": "Claude 2.0",
                "context_length": 100000,
                "capabilities": [
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT
                ]
            },
            {
                "id": "claude-instant-1.2",
                "name": "Claude Instant 1.2",
                "context_length": 100000,
                "capabilities": [
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT
                ]
            }
        ]
        
        return [
            ModelMetadata(
                id=model["id"],
                name=model["name"],
                provider=ModelProvider.ANTHROPIC,
                version=model["id"].split("-")[-1],
                status=ModelStatus.AVAILABLE,
                capabilities=model["capabilities"],
                parameters={
                    "context_length": model["context_length"],
                    "max_tokens": 4096,
                    "supports_vision": ModelCapability.VISION in model["capabilities"]
                }
            )
            for model in known_models
        ]
    
    async def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get details about a specific Anthropic model."""
        models = await self.list_models()
        for model in models:
            if model.id == model_id:
                return model
        return None
    
    async def generate_text(
        self,
        model_id: str,
        prompt: str,
        **kwargs
    ) -> str:
        """Generate text using the specified Anthropic model."""
        try:
            data = {
                "model": model_id,
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "max_tokens_to_sample": kwargs.get("max_tokens", 2000),
                "temperature": kwargs.get("temperature", 0.7),
                **{k: v for k, v in kwargs.items() if k in ["top_p", "top_k", "stop_sequences"]}
            }
            
            response = await self._request(
                "POST",
                "/v1/complete",
                json=data,
                timeout=300.0
            )
            
            return response.get("completion", "")
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
        """Generate a chat completion using the specified Anthropic model."""
        try:
            # Convert messages to Anthropic's format
            system_prompt = ""
            converted_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt += msg["content"] + "\n"
                else:
                    converted_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            data = {
                "model": model_id,
                "messages": converted_messages,
                "max_tokens": kwargs.get("max_tokens", 2000),
                "temperature": kwargs.get("temperature", 0.7),
                "system": system_prompt or None,
                **{k: v for k, v in kwargs.items() if k in ["top_p", "top_k", "stop_sequences"]}
            }
            
            response = await self._request(
                "POST",
                "/v1/messages",
                json=data,
                timeout=300.0
            )
            
            # Extract the assistant's message from the response
            if response and "content" in response and len(response["content"]) > 0:
                return "\n".join([block["text"] for block in response["content"] if block["type"] == "text"])
            return ""
        except Exception as e:
            error_msg = f"Failed to chat with model {model_id}: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
    
    async def close(self):
        """Clean up resources."""
        await self.client.aclose()
