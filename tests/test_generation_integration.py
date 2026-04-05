import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, Any
import dataclasses

from llm_mcp.tools.generation_tools import GenerationManager, GenerationConfig
from llm_mcp.services.provider_factory import ProviderFactory, _provider_factory
from llm_mcp.providers.base import BaseProvider


class MockProvider(BaseProvider):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self._name = "mock"

    async def generate(self, prompt: str, model: str, **kwargs):
        yield f"Mock response for {model}: {prompt}"

    async def chat(self, model_id, messages, **kwargs) -> str:
        return f"Mock chat response for {model_id}"

    async def list_models(self):
        return []

    async def pull_model(self, model_name: str):
        return {}

    async def get_model_info(self, model_name: str):
        return {}

    @property
    def name(self):
        return "mock"

    @property
    def supports_streaming(self):
        return True


@pytest.mark.asyncio
async def test_generation_manager_uses_provider():
    # Setup
    manager = GenerationManager()

    mock_provider = MockProvider()

    # Patch the global provider factory used by GenerationManager
    # Note: GenerationManager.__init__ binds self.provider_factory = _provider_factory

    with patch.object(
        manager.provider_factory, "get_provider_for_model", return_value=mock_provider
    ):
        # Test generate
        result = await manager.generate(model_id="test-model", prompt="Hello")
        assert result["text"] == "Mock response for test-model: Hello"
        assert result["finish_reason"] == "stop"
        assert result["model"] == "test-model"

        # Test chat
        result_chat = await manager.chat(
            model_id="test-model", messages=[{"role": "user", "content": "Hi"}]
        )
        assert result_chat["message"]["content"] == "Mock chat response for test-model"
        assert result_chat["model"] == "test-model"
