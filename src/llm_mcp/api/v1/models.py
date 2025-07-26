"""Pydantic models for the LLM MCP API."""
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class ProviderInfo(BaseModel):
    """Information about a provider."""
    name: str = Field(..., description="Name of the provider")
    description: str = Field(..., description="Description of the provider")
    capabilities: List[str] = Field(
        default_factory=list,
        description="List of capabilities supported by the provider"
    )


class ModelInfo(BaseModel):
    """Information about a model."""
    id: str = Field(..., description="Unique identifier for the model")
    name: str = Field(..., description="Name of the model")
    provider: str = Field(..., description="Name of the provider")
    description: Optional[str] = Field(
        None,
        description="Description of the model"
    )
    capabilities: List[str] = Field(
        default_factory=list,
        description="List of capabilities supported by the model"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Model-specific parameters and their descriptions"
    )


class GenerateRequest(BaseModel):
    """Request model for generating text."""
    prompt: str = Field(..., description="The input prompt")
    model: str = Field(..., description="The model to use for generation")
    provider: Optional[str] = Field(
        None,
        description="The provider to use. If not specified, the default provider will be used."
    )
    temperature: Optional[float] = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Higher values make output more random."
    )
    max_tokens: Optional[int] = Field(
        1024,
        gt=0,
        description="Maximum number of tokens to generate."
    )
    top_p: Optional[float] = Field(
        0.9,
        gt=0.0,
        le=1.0,
        description="Nucleus sampling parameter. Higher values mean more diversity."
    )
    frequency_penalty: Optional[float] = Field(
        0.0,
        ge=0.0,
        le=2.0,
        description="Penalty for using frequent tokens."
    )
    presence_penalty: Optional[float] = Field(
        0.0,
        ge=0.0,
        le=2.0,
        description="Penalty for using new tokens."
    )
    stop: Optional[List[str]] = Field(
        None,
        description="Sequences where the API will stop generating further tokens."
    )
    stream: bool = Field(
        False,
        description="Whether to stream the response as server-sent events."
    )


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    text: str = Field(..., description="The generated text")
    model: str = Field(..., description="The model used for generation")
    provider: str = Field(..., description="The provider used for generation")
    usage: Optional[Dict[str, int]] = Field(
        None,
        description="Information about token usage"
    )


class ModelStatus(str, Enum):
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
    message: Optional[str] = Field(None, description="Additional information")
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional details about the operation"
    )
