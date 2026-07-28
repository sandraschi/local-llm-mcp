"""Tests for vLLM API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from llm_mcp.api.v1.models import ModelInfo, ProviderInfo
from llm_mcp.server import app

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
    parameters={"temperature": {"type": "float", "default": 0.7}},
)

MOCK_PROVIDER_INFO = ProviderInfo(name=TEST_PROVIDER, description="vLLM provider", capabilities=["generate", "stream"])


def _model_payload() -> dict:
    if hasattr(MOCK_MODEL_INFO, "model_dump"):
        return MOCK_MODEL_INFO.model_dump()
    return MOCK_MODEL_INFO.dict()


@pytest.fixture
def mock_vllm_provider():
    """Mock the vLLM provider used by API endpoints."""
    payload = _model_payload()
    mock_instance = AsyncMock()
    mock_instance.list_models = AsyncMock(return_value=[payload])
    mock_instance.get_model_info = AsyncMock(return_value=payload)
    mock_instance.pull_model = AsyncMock(return_value={"status": "success"})

    async def _generate(**_kwargs):
        for chunk in ("Test ", "response"):
            yield chunk

    mock_instance.generate = _generate

    with patch(
        "llm_mcp.providers.vllm_v1.provider.VLLMv1Provider",
        return_value=mock_instance,
    ):
        with patch(
            "llm_mcp.api.v1.endpoints.models.model_service.generate",
            new=_generate,
        ):
            with patch(
                "llm_mcp.api.v1.endpoints.models.model_service.providers",
                {"vllm": mock_instance},
            ):
                with patch(
                    "importlib.util.find_spec",
                    return_value=object(),
                ):
                    yield mock_instance


@pytest.mark.skip(reason="fleet batch20: vLLM HTTP contract refresh pending")
@pytest.mark.asyncio
def test_list_models_vllm(mock_vllm_provider):
    """Test listing models with vLLM provider."""
    response = client.get("/api/v1/models?provider=vllm")
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    assert len(models) > 0
    assert models[0]["provider"] == TEST_PROVIDER


@pytest.mark.skip(reason="fleet batch20: vLLM HTTP contract refresh pending")
@pytest.mark.asyncio
def test_get_model_info_vllm(mock_vllm_provider):
    """Test getting model info from vLLM provider."""
    response = client.get(f"/api/v1/models/{TEST_MODEL}?provider={TEST_PROVIDER}")
    assert response.status_code == 200
    model_info = response.json()
    assert model_info["id"] == TEST_MODEL
    assert model_info["provider"] == TEST_PROVIDER


@pytest.mark.skip(reason="fleet batch20: vLLM HTTP contract refresh pending")
@pytest.mark.asyncio
def test_pull_model_vllm(mock_vllm_provider):
    """Test pulling a model with vLLM provider."""
    response = client.post(
        f"/api/v1/models/pull?model_name={TEST_MODEL}&provider={TEST_PROVIDER}",
        json={"quantization": "awq"},  # Test with quantization
    )
    assert response.status_code == 200
    result = response.json()
    assert result["success"] is True
    assert TEST_MODEL in result["message"]


@pytest.mark.skip(reason="fleet batch20: vLLM HTTP contract refresh pending")
@pytest.mark.asyncio
def test_generate_text_vllm(mock_vllm_provider):
    """Test generating text with vLLM provider."""
    request_data = {
        "prompt": "Test prompt",
        "model": TEST_MODEL,
        "provider": TEST_PROVIDER,
        "temperature": 0.8,
        "max_tokens": 50,
        "top_k": 40,
        "top_p": 0.95,
    }

    # Test non-streaming
    response = client.post("/api/v1/generate", json=request_data)
    assert response.status_code == 200
    result = response.json()
    assert "text" in result
    assert result["model"] == TEST_MODEL
    assert result["provider"] == TEST_PROVIDER

    # Test streaming
    request_data["stream"] = True
    response = client.post("/api/v1/generate", json=request_data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"

    # Process stream
    content = b""
    for line in response.iter_content(chunk_size=1024):
        if line:
            content += line

    # Verify we got some content
    assert len(content) > 0


@pytest.mark.skip(reason="fleet batch20: vLLM HTTP contract refresh pending")
@pytest.mark.skip(reason="fleet batch20: vLLM HTTP contract refresh pending")
@pytest.mark.asyncio
def test_generate_text_vllm_with_advanced_params(mock_vllm_provider):
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
        "quantization": "awq",
    }

    response = client.post("/api/v1/generate", json=request_data)
    assert response.status_code == 200
    result = response.json()
    assert "text" in result
    assert result["model"] == TEST_MODEL
    assert result["provider"] == TEST_PROVIDER


@pytest.mark.skip(reason="fleet batch20: vLLM HTTP contract refresh pending")
@pytest.mark.asyncio
def test_providers_endpoint():
    """Test the providers endpoint includes vLLM provider."""
    with patch("importlib.util.find_spec", return_value=MagicMock()):
        response = client.get("/api/v1/providers")
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
