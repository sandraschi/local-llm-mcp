"""Pydantic models for the LLM MCP API."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ProviderInfo(BaseModel):
    """Information about a provider."""

    name: str = Field(..., description="Name of the provider")
    description: str = Field(..., description="Description of the provider")
    capabilities: list[str] = Field(default_factory=list, description="List of capabilities supported by the provider")
    parameters: dict[str, Any] | None = Field(
        default=None, description="Provider-specific parameters and their descriptions"
    )


class ModelIntelligence(BaseModel):
    """Rich metadata about a model's intelligence and lineage."""

    hf_id: str | None = Field(None, description="Hugging Face Model ID")
    developer: str | None = Field(None, description="The entity that developed the model")
    release_date: str | None = Field(None, description="Release date in YYYY-MM-DD")
    strengths: list[str] = Field(default_factory=list, description="What the model excels at")
    weaknesses: list[str] = Field(default_factory=list, description="Known limitations")
    best_for: str | None = Field(None, description="Recommended use case")
    is_legacy: bool = Field(False, description="Whether this is a pre-2025 legacy model")
    vram_required_gb: float | None = Field(None, description="Estimated VRAM required in GB")
    quantization_info: str | None = Field(None, description="Info about common quantizations (GGUF, EXL2, etc.)")
    model_card_url: str | None = Field(None, description="Direct link to model card/docs")


class ModelInfo(BaseModel):
    """Information about a model."""

    id: str = Field(..., description="Unique identifier for the model")
    name: str = Field(..., description="Name of the model")
    provider: str = Field(..., description="Name of the provider")
    description: str | None = Field(default=None, description="Description of the model")
    capabilities: list[str] = Field(default_factory=list, description="List of capabilities supported by the model")
    parameters: dict[str, Any] | None = Field(
        default=None, description="Model-specific parameters and their descriptions"
    )
    intelligence: ModelIntelligence | None = Field(
        default=None, description="Rich intelligence and metadata about the model"
    )


class GenerateRequest(BaseModel):
    """Request model for generating text."""

    prompt: str = Field(..., description="The input prompt")
    model: str = Field(..., description="The model to use for generation")
    provider: str | None = Field(
        None, description="The provider to use. If not specified, the default provider will be used."
    )
    images: list[str] | None = Field(
        None, description="Optional list of base64-encoded images for multimodal vision tasks."
    )
    temperature: float | None = Field(
        0.7, ge=0.0, le=2.0, description="Sampling temperature. Higher values make output more random."
    )
    max_tokens: int | None = Field(100, ge=1, description="Maximum number of tokens to generate.")
    top_p: float | None = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Nucleus sampling: only the smallest set of most probable tokens with "
            "probabilities that add up to top_p or higher are kept."
        ),
    )
    frequency_penalty: float | None = Field(
        0.0,
        ge=-2.0,
        le=2.0,
        description=(
            "Penalty for token frequency. Positive values penalize new tokens based "
            "on their frequency in the text so far."
        ),
    )
    presence_penalty: float | None = Field(
        0.0,
        ge=-2.0,
        le=2.0,
        description=(
            "Penalty for new tokens. Positive values penalize new tokens based on "
            "whether they appear in the text so far."
        ),
    )
    stop: list[str] | None = Field(
        None, description="List of sequences where the API will stop generating further tokens."
    )
    stream: bool = Field(False, description="Whether to stream the response as server-sent events.")
    # vLLM specific parameters
    top_k: int | None = Field(
        None,
        ge=-1,
        description="The number of highest probability vocabulary tokens to keep for top-k filtering.",
    )
    best_of: int | None = Field(
        None, ge=1, description="Generates best_of completions server-side and returns the best."
    )
    use_beam_search: bool = Field(False, description="Whether to use beam search instead of sampling.")
    length_penalty: float | None = Field(
        1.0, ge=0.0, description="Exponential penalty to the length that is used with beam search."
    )
    early_stopping: bool | None = Field(
        False,
        description=(
            "Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not."
        ),
    )
    stop_token_ids: list[int] | None = Field(
        None, description="List of token IDs where the API will stop generating further tokens."
    )
    ignore_eos: bool = Field(False, description="Whether to ignore the EOS token and continue generating.")
    logprobs: int | None = Field(
        None, ge=0, le=5, description="Number of most likely tokens to return at each token position."
    )
    prompt_logprobs: int | None = Field(
        None, ge=0, le=5, description="Number of most likely tokens to return for each prompt token."
    )
    # vLLM engine configuration
    tensor_parallel_size: int | None = Field(None, ge=1, description="Number of GPUs to use for distributed execution.")
    gpu_memory_utilization: float | None = Field(None, ge=0.0, le=1.0, description="Fraction of GPU memory to use.")
    max_seq_len: int | None = Field(None, ge=1, description="Maximum sequence length.")
    quantization: str | None = Field(
        None, description="Quantization method to use (e.g., 'awq', 'gptq', 'squeezellm')."
    )


class GenerateResponse(BaseModel):
    """Response model for text generation."""

    text: str = Field(..., description="The generated text")
    model: str = Field(..., description="The model used for generation")
    provider: str = Field(..., description="The provider used for generation")
    usage: dict[str, int] | None = Field(None, description="Information about token usage")


class ModelStatus(StrEnum):
    """Status of a model."""

    PENDING = "pending"
    DOWNLOADING = "downloading"
    READY = "ready"
    ERROR = "error"


class ModelOperationResponse(BaseModel):
    """Response model for model operations."""

    model: str = Field(..., description="Name of the model")
    provider: str = Field(..., description="Name of the provider")
    status: ModelStatus = Field(..., description="Status of the operation")
    message: str | None = Field(None, description="Additional information")
    details: dict[str, Any] | None = Field(None, description="Additional details about the operation")
