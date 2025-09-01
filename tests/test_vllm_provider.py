"""Tests for the vLLM provider."""

import asyncio
import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

from llm_mcp.providers.vllm_v1.provider import VLLMv1Provider, VLLMv1Config
from llm_mcp.providers.base import BaseProvider

# Skip these tests if vLLM is not installed
pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_VLLM", "0").lower() in ("1", "true", "yes"),
    reason="vLLM tests are disabled by default. Set TEST_VLLM=1 to enable."
)

# Test configuration
TEST_CONFIG = {
    "model": "gpt2",  # Small model for testing
    "max_seq_len": 256,
    "gpu_memory_utilization": 0.5,
}

@pytest.fixture
def vllm_provider():
    """Fixture that provides a vLLM provider instance for testing."""
    return VLLMv1Provider(TEST_CONFIG)

@pytest.mark.asyncio
async def test_vllm_provider_initialization(vllm_provider):
    """Test that the vLLM provider initializes correctly."""
    assert isinstance(vllm_provider, BaseProvider)
    assert vllm_provider.name == "vllm_v1"
    assert not vllm_provider.is_ready

@pytest.mark.asyncio
async def test_vllm_initialize(vllm_provider):
    """Test that the vLLM provider can be initialized."""
    await vllm_provider.initialize()
    assert vllm_provider.is_ready
    assert vllm_provider._model_loaded

@pytest.mark.asyncio
async def test_list_models(vllm_provider):
    """Test that the provider can list available models."""
    await vllm_provider.initialize()
    models = await vllm_provider.list_models()
    assert isinstance(models, list)
    assert len(models) > 0
    assert "id" in models[0]
    assert "name" in models[0]
    assert "description" in models[0]

@pytest.mark.asyncio
async def test_generate_text(vllm_provider):
    """Test that the provider can generate text."""
    await vllm_provider.initialize()
    
    prompt = "The quick brown fox"
    full_response = ""
    
    async for chunk in vllm_provider.generate(prompt, max_tokens=10):
        full_response += chunk
    
    assert len(full_response) > 0
    assert isinstance(full_response, str)

@pytest.mark.asyncio
async def test_pull_model(vllm_provider):
    """Test that the provider can pull a model."""
    # Clean up first
    if vllm_provider.is_ready:
        await vllm_provider.cleanup()
    
    model_name = "gpt2"
    model_info = await vllm_provider.pull_model(model_name)
    
    assert model_info["id"] == model_name
    assert vllm_provider.is_ready
    assert vllm_provider._model_loaded

@pytest.mark.asyncio
async def test_health_check(vllm_provider):
    """Test the health check functionality."""
    health = await vllm_provider.health_check()
    assert isinstance(health, dict)
    assert "status" in health
    
    # Should be unhealthy before initialization
    assert health["status"] == "unhealthy"
    
    # Should be healthy after initialization
    await vllm_provider.initialize()
    health = await vllm_provider.health_check()
    assert health["status"] == "healthy"

@pytest.mark.asyncio
async def test_metrics(vllm_provider):
    """Test that metrics are collected correctly."""
    await vllm_provider.initialize()
    
    # Generate some text to update metrics
    prompt = "Test prompt for metrics"
    async for _ in vllm_provider.generate(prompt, max_tokens=5):
        pass
    
    metrics = await vllm_provider.get_metrics()
    
    assert isinstance(metrics, dict)
    assert metrics["total_requests"] > 0
    assert metrics["successful_requests"] > 0
    assert metrics["total_tokens_generated"] > 0

@pytest.mark.asyncio
async def test_cleanup(vllm_provider):
    """Test that cleanup releases resources."""
    await vllm_provider.initialize()
    assert vllm_provider.is_ready
    
    await vllm_provider.cleanup()
    assert not vllm_provider.is_ready
    assert not vllm_provider._model_loaded

# Mock tests for when vLLM is not available
@patch('llm_mcp.providers.vllm_v1.provider.VLLM_AVAILABLE', False)
def test_vllm_not_available():
    """Test that an error is raised when vLLM is not available."""
    with pytest.raises(ImportError):
        VLLMv1Provider({})

# Test with different configurations
@pytest.mark.parametrize("config", [
    {"model": "gpt2", "tensor_parallel_size": 1},
    {"model": "gpt2", "gpu_memory_utilization": 0.3},
    {"model": "gpt2", "quantization": "awq"},
])
@pytest.mark.asyncio
async def test_different_configs(config):
    """Test the provider with different configurations."""
    provider = VLLMv1Provider(config)
    try:
        await provider.initialize()
        assert provider.is_ready
        
        # Test a simple generation
        async for _ in provider.generate("Test prompt", max_tokens=5):
            pass
            
    finally:
        await provider.cleanup()
