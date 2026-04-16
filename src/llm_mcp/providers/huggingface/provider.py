"""Hugging Face provider implementation for LLM MCP."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)

# Optional diffusion imports
try:
    from diffusers import (
        DiffusionPipeline,
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
    )

    DIFFUSERS_AVAILABLE = True
except (ImportError, RuntimeError):
    DIFFUSERS_AVAILABLE = False

from ..base import BaseProvider
from .config import HuggingFaceConfig

logger = logging.getLogger(__name__)


class HuggingFaceProvider(BaseProvider):
    """Hugging Face provider for LLM MCP.

    This provider supports both local and remote Hugging Face models,
    with optimizations for GPU inference and batching.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize the Hugging Face provider.

        Args:
            config: Configuration dictionary with Hugging Face settings
        """
        super().__init__(config)
        self.config = HuggingFaceConfig(**config)
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._diffusion_pipeline = None  # For diffusion models
        self._device = self._get_device()
        self._is_initialized = False

    async def initialize(self):
        """Initialize the provider and load the model."""
        if self._is_initialized:
            return

        logger.info(f"Initializing Hugging Face provider with model: {self.config.model_name}")

        # Load model and tokenizer in a separate thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()

        # Determine if this is a diffusion model
        if self._is_diffusion_model():
            await loop.run_in_executor(None, self._load_diffusion_model)
        else:
            await loop.run_in_executor(None, self._load_model)

        self._is_initialized = True
        logger.info("Hugging Face provider initialized successfully")

    def _load_model(self):
        """Load the model and tokenizer."""

        model_kwargs = {
            "revision": self.config.model_revision,
            "trust_remote_code": self.config.trust_remote_code,
            **self.config.model_kwargs,
        }

        # Determine model class based on task
        if self.config.task == "text-generation":
            model_class = AutoModelForCausalLM
        elif self.config.task == "text2text-generation":
            model_class = AutoModelForSeq2SeqLM
        else:
            model_class = AutoModelForCausalLM  # Default

        # Load model and tokenizer
        self._model = model_class.from_pretrained(self.config.model_name, **model_kwargs).to(self._device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            revision=self.config.model_revision,
            trust_remote_code=self.config.trust_remote_code,
        )

        # Initialize pipeline if needed
        if self.config.task:
            self._pipeline = pipeline(
                self.config.task,
                model=self._model,
                tokenizer=self._tokenizer,
                device=self._device,
                **self.config.pipeline_kwargs,
            )

    def _is_diffusion_model(self) -> bool:
        """Check if the configured model is a diffusion model."""
        diffusion_keywords = [
            "stable-diffusion",
            "sdxl",
            "flux",
            "diffusion",
            "latent-diffusion",
            "dreamlike",
            "anything",
            "realistic-vision",
            "openjourney",
        ]
        model_name_lower = self.config.model_name.lower()
        return any(keyword in model_name_lower for keyword in diffusion_keywords)

    def _load_diffusion_model(self):
        """Load a diffusion model."""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers not available. Install with: pip install diffusers")

        logger.info(f"Loading diffusion model: {self.config.model_name}")

        # Determine pipeline type
        model_name_lower = self.config.model_name.lower()
        if "sdxl" in model_name_lower:
            pipeline_class = StableDiffusionXLPipeline
        else:
            pipeline_class = StableDiffusionPipeline

        # Load diffusion pipeline
        self._diffusion_pipeline = pipeline_class.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            **self.config.model_kwargs,
        )

        # Apply optimizations
        if hasattr(self._diffusion_pipeline, "enable_attention_slicing"):
            self._diffusion_pipeline.enable_attention_slicing()
        if hasattr(self._diffusion_pipeline, "enable_vae_slicing"):
            self._diffusion_pipeline.enable_vae_slicing()

        self._diffusion_pipeline.to(self._device)
        logger.info("Diffusion model loaded successfully")

    def _get_device(self) -> str:
        """Determine the best device to use."""
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device

    async def list_models(self) -> list[dict[str, Any]]:
        """List available models from the Hugging Face Hub.

        Note: This is a placeholder implementation. In a real implementation,
        you would fetch available models from the Hugging Face Hub API.
        """
        return [
            {
                "id": self.config.model_name,
                "name": self.config.model_name.split("/")[-1],
                "description": f"Hugging Face model: {self.config.model_name}",
                "capabilities": [self.config.task] if self.config.task else [],
            }
        ]

    async def generate(self, prompt: str, model: str | None = None, **kwargs) -> AsyncGenerator[str, None]:
        """Generate a response from the model.

        Args:
            prompt: The input prompt
            model: Model to use (overrides the one in config if provided)
            **kwargs: Additional generation parameters

        Yields:
            Chunks of the generated response
        """
        if not self._is_initialized:
            await self.initialize()

        # Use the specified model or the default one
        model_name = model or self.config.model_name
        if model_name != self.config.model_name:
            # TODO: Handle model switching if needed
            pass

        # Prepare generation parameters
        gen_kwargs = {
            "max_length": kwargs.get("max_length", 100),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 50),
            "do_sample": kwargs.get("do_sample", True),
            **{
                k: v for k, v in kwargs.items() if k not in ["max_length", "temperature", "top_p", "top_k", "do_sample"]
            },
        }

        if self._pipeline:
            # Use pipeline for generation
            result = await asyncio.get_event_loop().run_in_executor(None, lambda: self._pipeline(prompt, **gen_kwargs))

            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    yield result[0]["generated_text"]
                else:
                    yield str(result[0])
            else:
                yield str(result)
        else:
            # Fallback to manual generation
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    **gen_kwargs,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

                generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
                yield generated_text

    async def pull_model(self, model_name: str) -> dict[str, Any]:
        """Download a model from Hugging Face Hub.

        Args:
            model_name: Name of the model to download

        Returns:
            Dictionary with download status and model info
        """
        from huggingface_hub import model_info, snapshot_download

        try:
            # Get model info
            info = model_info(model_name, token=self.config.api_key)

            # Download the model
            local_path = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: snapshot_download(
                    repo_id=model_name,
                    revision=self.config.model_revision,
                    token=self.config.api_key,
                    local_files_only=False,
                    resume_download=True,
                ),
            )

            return {
                "status": "success",
                "model_name": model_name,
                "local_path": local_path,
                "model_info": {
                    "id": info.id,
                    "pipeline_tag": info.pipeline_tag,
                    "tags": info.tags,
                    "downloads": info.downloads,
                },
            }

        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e!s}")
            return {"status": "error", "model_name": model_name, "error": str(e)}

    async def get_model_info(self, model_name: str | None = None) -> dict[str, Any]:
        """Get information about a model.

        Args:
            model_name: Name of the model (defaults to the configured model)

        Returns:
            Dictionary with model information
        """
        if not model_name:
            model_name = self.config.model_name

        try:
            from huggingface_hub import model_info

            info = model_info(model_name, token=self.config.api_key)

            return {
                "id": info.id,
                "pipeline_tag": info.pipeline_tag,
                "tags": info.tags,
                "downloads": info.downloads,
                "last_modified": info.last_modified.isoformat() if info.last_modified else None,
                "model_size": info.safetensors.get("total") if info.safetensors else None,
                "license": info.cardData.get("license") if hasattr(info, "cardData") else None,
                "model_type": info.config.get("model_type") if hasattr(info, "config") else None,
                "architectures": info.config.get("architectures") if hasattr(info, "config") else None,
            }

        except Exception as e:
            logger.error(f"Failed to get info for model {model_name}: {e!s}")
            return {"status": "error", "model_name": model_name, "error": str(e)}

    async def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        **kwargs,
    ) -> dict[str, Any]:
        """Generate image using diffusion model.

        Args:
            prompt: Text prompt for image generation
            negative_prompt: Negative prompt
            width: Image width
            height: Image height
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            **kwargs: Additional generation parameters

        Returns:
            Generated image data
        """
        if not self._diffusion_pipeline:
            return {"error": "Diffusion model not loaded. Ensure model_name is a diffusion model."}

        if not DIFFUSERS_AVAILABLE:
            return {"error": "Diffusers not available. Install with: pip install diffusers"}

        try:
            # Generate image
            result = self._diffusion_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                **kwargs,
            )

            image = result.images[0]

            # Convert to base64
            import base64
            from io import BytesIO

            buffer = BytesIO()
            image.save(buffer, format="PNG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode()

            return {
                "image_base64": image_base64,
                "model": self.config.model_name,
                "prompt": prompt,
                "width": width,
                "height": height,
                "inference_steps": num_inference_steps,
            }

        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return {"error": str(e)}
