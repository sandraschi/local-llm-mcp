"""Tests for vLLM API endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
from typing import Dict, Any, List, Optional
import json

from src.llm_mcp.main import app
from src.llm_mcp.api.v1.models import GenerateRequest, ModelInfo, ProviderInfo

# Test client
client = TestClient(app)

# Test data
TEST_MODEL = "gpt2"  # Small model for testing
TEST_PROVIDER = "vllm"

# Mock responses
MOCK_MODEL_INFO = ModelInfo(
    id=TEST_MODEL,
    name=TEST_MODEL,
    provider=TEST_PROVIDER,
    description="Test model",
    capabilities=["generate"],
    parameters={"temperature": {"type": "float", "default": 0.7}}
)

MOCK_PROVIDER_INFO = ProviderInfo(
    name=TEST_PROVIDER,
    description="vLLM provider",
    capabilities=["generate", "stream"]
)

@pytest.fixture
def mock_vllm_provider():
    """Mock the vLLM provider."""
    with patch('src.llm_mcp.services.model_service.VLLMv1Provider') as mock_provider:
        # Mock provider methods
        mock_instance = mock_provider.return_value
        mock_instance.initialize = AsyncMock()
        mock_instance.list_models = AsyncMock(return_value=[MOCK_MODEL.dict()])
        mock_instance.get_model_info = AsyncMock(return_value=MOCK_MODEL.dict())
        mock_instance.generate = AsyncMock(return_value=AsyncMock(__aiter__=lambda self: iter(["Test ", "response"])))
        mock_instance.pull_model = AsyncMock(return_value={"status": "success"})
        
        # Add to providers
        with patch.dict('src.llm_mcp.services.model_service.PROVIDER_CLASSES', 
                       {'vllm': 'src.llm_mcp.providers.vllm_v1.provider.VLLMv1Provider'}, clear=False):
            yield mock_instance

@pytest.mark.asyncio
async def test_list_models_vllm(mock_vllm_provider):
    """Test listing models with vLLM provider."""
    response = client.get("/v1/models?provider=vllm")
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    assert len(models) > 0
    assert models[0]["provider"] == TEST_PROVIDER

@pytest.mark.asyncio
async def test_get_model_info_vllm(mock_vllm_provider):
    """Test getting model info from vLLM provider."""
    response = client.get(f"/v1/models/{TEST_MODEL}?provider={TEST_PROVIDER}")
    assert response.status_code == 200
    model_info = response.json()
    assert model_info["id"] == TEST_MODEL
    assert model_info["provider"] == TEST_PROVIDER

@pytest.mark.asyncio
async def test_pull_model_vllm(mock_vllm_provider):
    """Test pulling a model with vLLM provider."""
    response = client.post(
        f"/v1/models/pull?model={TEST_MODEL}&provider={TEST_PROVIDER}",
        json={"quantization": "awq"}  # Test with quantization
    )
    assert response.status_code == 200
    result = response.json()
    assert result["success"] is True
    assert TEST_MODEL in result["message"]

@pytest.mark.asyncio
async def test_generate_text_vllm(mock_vllm_provider):
    """Test generating text with vLLM provider."""
    request_data = {
        "prompt": "Test prompt",
        "model": TEST_MODEL,
        "provider": TEST_PROVIDER,
        "temperature": 0.8,
        "max_tokens": 50,
        "top_k": 40,
        "top_p": 0.95
    }
    
    # Test non-streaming
    response = client.post("/v1/generate", json=request_data)
    assert response.status_code == 200
    result = response.json()
    assert "text" in result
    assert result["model"] == TEST_MODEL
    assert result["provider"] == TEST_PROVIDER
    
    # Test streaming
    request_data["stream"] = True
    response = client.post("/v1/generate", json=request_data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    
    # Process stream
    content = b""
    for line in response.iter_content(chunk_size=1024):
        if line:
            content += line
    
    # Verify we got some content
    assert len(content) > 0

@pytest.mark.asyncio
async def test_generate_text_vllm_with_advanced_params(mock_vllm_provider):
    """Test generating text with advanced vLLM parameters."""
    request_data = {
        "prompt": "Test prompt with advanced params",
        "model": TEST_MODEL,
        "provider": TEST_PROVIDER,
        "temperature": 0.7,
        "max_tokens": 100,
        "top_k": 50,
        "top_p": 0.9,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.5,
        "best_of": 3,
        "use_beam_search": False,
        "length_penalty": 1.0,
        "stop": ["\n"],
        "stop_token_ids": [50256],
        "ignore_eos": False,
        "logprobs": 1,
        "prompt_logprobs": 1,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "max_seq_len": 2048,
        "quantization": "awq"
    }
    
    response = client.post("/v1/generate", json=request_data)
    assert response.status_code == 200
    result = response.json()
    assert "text" in result
    assert result["model"] == TEST_MODEL
    assert result["provider"] == TEST_PROVIDER

@pytest.mark.asyncio
async def test_providers_endpoint():
    """Test the providers endpoint includes vLLM provider."""
    response = client.get("/v1/providers")
    assert response.status_code == 200
    providers = response.json()
    assert isinstance(providers, list)
    
    # Check if vLLM provider is in the list
    vllm_providers = [p for p in providers if p.get("name") in ("vllm", "vllm_v1")]
    assert len(vllm_providers) > 0, "vLLM provider not found in providers list"
    
    # Check capabilities
    vllm_provider = vllm_providers[0]
    assert "generate" in vllm_provider.get("capabilities", [])
    assert "stream" in vllm_provider.get("capabilities", [])
    
    # Check parameters
    assert "parameters" in vllm_provider
    assert "model" in vllm_provider["parameters"]
    assert "tensor_parallel_size" in vllm_provider["parameters"]
    assert "gpu_memory_utilization" in vllm_provider["parameters"]
    assert "quantization" in vllm_provider["parameters"]
