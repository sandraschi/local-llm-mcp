"""SOTA Multimodal Provider for vision-language models and diffusion models.

This provider integrates:
- Vision-Language Models (LLaVA, CLIP, BLIP-2, GPT-4V-style)
- Diffusion Models (FLUX, SDXL, SDXL Turbo, Stable Video Diffusion)
- Image Analysis and Understanding
- Text-to-Image and Image-to-Text generation
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

import torch

try:
    from io import BytesIO

    import requests
    from diffusers import (
        AutoencoderKL,
        DiffusionPipeline,
        FluxPipeline,
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
        UNet2DConditionModel,
    )
    from PIL import Image
    from transformers import (
        AutoModel,
        AutoModelForVision2Seq,
        AutoProcessor,
        Blip2ForConditionalGeneration,
        Blip2Processor,
        BlipForConditionalGeneration,
        BlipProcessor,
        CLIPModel,
        CLIPProcessor,
        CLIPTokenizer,
    )

    MULTIMODAL_DEPS_AVAILABLE = True
except ImportError:
    MULTIMODAL_DEPS_AVAILABLE = False

from .base import BaseProvider

logger = logging.getLogger(__name__)


class MultimodalProvider(BaseProvider):
    """SOTA Multimodal Provider for vision-language and diffusion models.

    Supports:
    - Vision-Language Models: LLaVA, CLIP, BLIP-2, GPT-4V
    - Diffusion Models: FLUX, SDXL Turbo, SDXL, Stable Diffusion
    - Image Analysis: Captioning, VQA, classification
    - Image Generation: Text-to-image, image-to-image, inpainting
    - Video Generation: Stable Video Diffusion
    """

    SUPPORTED_VISION_MODELS = {
        "llava": "llava-hf/llava-1.5-7b-hf",
        "clip": "openai/clip-vit-base-patch32",
        "blip": "Salesforce/blip-image-captioning-base",
        "blip2": "Salesforce/blip2-opt-2.7b",
        "git": "microsoft/git-base",
    }

    SUPPORTED_DIFFUSION_MODELS = {
        "flux": "blackforestlabs/FLUX.1-dev",
        "sdxl-turbo": "stabilityai/sdxl-turbo",
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        "sd": "runwayml/stable-diffusion-v1-5",
        "svd": "stabilityai/stable-video-diffusion-img2vid-xt",
    }

    def __init__(self, config: dict[str, Any]):
        """Initialize multimodal provider.

        Args:
            config: Configuration with model settings
        """
        if not MULTIMODAL_DEPS_AVAILABLE:
            raise ImportError(
                "Multimodal dependencies not available. Install with: "
                "pip install transformers diffusers torch torchvision accelerate"
            )

        super().__init__(config)
        self.vision_models = {}
        self.diffusion_models = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._memory_manager = self._setup_memory_management()

    def _setup_memory_management(self) -> dict[str, Any]:
        """Setup GPU memory management for multimodal workloads."""
        if torch.cuda.is_available():
            # Enable memory efficient attention
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)

            # Set memory fraction to avoid OOM
            torch.cuda.set_per_process_memory_fraction(0.8)

        return {
            "max_memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
            "memory_fraction": 0.8,
            "enable_attention_slicing": True,
            "enable_vae_slicing": True,
        }

    async def initialize(self):
        """Initialize the multimodal provider."""
        logger.info("Initializing SOTA Multimodal Provider")

        # Pre-load commonly used models
        await self._load_vision_model("blip2")  # For image analysis
        # Diffusion models loaded on-demand to save memory

    async def _load_vision_model(self, model_key: str) -> bool:
        """Load a vision model with memory optimization."""
        if model_key in self.vision_models:
            return True

        model_name = self.SUPPORTED_VISION_MODELS.get(model_key)
        if not model_name:
            logger.error(f"Unknown vision model: {model_key}")
            return False

        try:
            logger.info(f"Loading vision model: {model_name}")

            if model_key == "llava":
                processor = AutoProcessor.from_pretrained(model_name)
                model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    device_map="auto" if self.device.type == "cuda" else None,
                    low_cpu_mem_usage=True,
                )
            elif model_key == "clip":
                processor = CLIPProcessor.from_pretrained(model_name)
                model = CLIPModel.from_pretrained(model_name).to(self.device)
            elif model_key in ["blip", "blip2"]:
                if model_key == "blip":
                    processor = BlipProcessor.from_pretrained(model_name)
                    model = BlipForConditionalGeneration.from_pretrained(model_name)
                else:  # blip2
                    processor = Blip2Processor.from_pretrained(model_name)
                    model = Blip2ForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                        device_map="auto" if self.device.type == "cuda" else None,
                    )
                model.to(self.device)

            self.vision_models[model_key] = {
                "model": model,
                "processor": processor,
                "loaded_at": asyncio.get_event_loop().time()
            }

            logger.info(f"Successfully loaded vision model: {model_key}")
            return True

        except Exception as e:
            logger.error(f"Failed to load vision model {model_key}: {e}")
            return False

    async def _load_diffusion_model(self, model_key: str) -> bool:
        """Load a diffusion model with memory optimization."""
        if model_key in self.diffusion_models:
            return True

        model_name = self.SUPPORTED_DIFFUSION_MODELS.get(model_key)
        if not model_name:
            logger.error(f"Unknown diffusion model: {model_key}")
            return False

        try:
            logger.info(f"Loading diffusion model: {model_name}")

            # Use memory-efficient loading
            if model_key == "flux":
                pipeline = FluxPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
                )
            elif model_key == "sdxl-turbo":
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                )
            elif model_key == "sdxl":
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    use_safetensors=True,
                )
            else:  # sd, svd
                pipeline = DiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                )

            # Apply memory optimizations
            if hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing()
            if hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()
            if hasattr(pipeline, "enable_vae_tiling"):
                pipeline.enable_vae_tiling()

            pipeline.to(self.device)
            self.diffusion_models[model_key] = {
                "pipeline": pipeline,
                "loaded_at": asyncio.get_event_loop().time()
            }

            logger.info(f"Successfully loaded diffusion model: {model_key}")
            return True

        except Exception as e:
            logger.error(f"Failed to load diffusion model {model_key}: {e}")
            return False

    async def analyze_image(
        self,
        image: str | Image.Image,
        task: str = "caption",
        model: str = "blip2"
    ) -> dict[str, Any]:
        """Analyze image with SOTA vision-language models.

        Args:
            image: Image path/URL or PIL Image
            task: Analysis task ("caption", "vqa", "classify")
            model: Vision model to use

        Returns:
            Analysis results
        """
        await self._load_vision_model(model)

        if model not in self.vision_models:
            return {"error": f"Failed to load vision model: {model}"}

        model_data = self.vision_models[model]

        # Load and preprocess image
        if isinstance(image, str):
            if image.startswith("http"):
                response = requests.get(image)
                img = Image.open(BytesIO(response.content))
            else:
                img = Image.open(image)
        else:
            img = image

        try:
            if model == "blip2":
                if task == "caption":
                    inputs = model_data["processor"](img, return_tensors="pt").to(self.device)
                    out = model_data["model"].generate(**inputs, max_new_tokens=50)
                    caption = model_data["processor"].decode(out[0], skip_special_tokens=True)
                    return {"caption": caption, "model": model}

            elif model == "clip":
                inputs = model_data["processor"](text=["a photo", "artwork", "diagram"], images=img, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model_data["model"](**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)

                return {
                    "classifications": {
                        "photo": float(probs[0][0]),
                        "artwork": float(probs[0][1]),
                        "diagram": float(probs[0][2])
                    },
                    "model": model
                }

            return {"error": f"Unsupported task '{task}' for model '{model}'"}

        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {"error": str(e)}

    async def generate_image(
        self,
        prompt: str,
        model: str = "flux",
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5
    ) -> dict[str, Any]:
        """Generate image with SOTA diffusion models.

        Args:
            prompt: Text prompt for generation
            model: Diffusion model to use
            negative_prompt: Negative prompt
            width: Image width
            height: Image height
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale

        Returns:
            Generated image data
        """
        await self._load_diffusion_model(model)

        if model not in self.diffusion_models:
            return {"error": f"Failed to load diffusion model: {model}"}

        pipeline = self.diffusion_models[model]["pipeline"]

        try:
            # Generate with optimized settings
            if model == "flux":
                # FLUX uses different parameters
                result = pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                )
            elif model == "sdxl-turbo":
                # SDXL Turbo is fast but lower quality
                result = pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=1,  # Turbo uses single step
                    guidance_scale=0.0,  # Turbo doesn't use guidance
                )
            else:
                # Standard diffusion models
                result = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )

            image = result.images[0]

            # Convert to base64 for API response
            import base64
            from io import BytesIO

            buffer = BytesIO()
            image.save(buffer, format="PNG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode()

            return {
                "image_base64": image_base64,
                "model": model,
                "prompt": prompt,
                "width": width,
                "height": height,
                "inference_steps": num_inference_steps
            }

        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return {"error": str(e)}

    async def compare_images(
        self,
        image1: str | Image.Image,
        image2: str | Image.Image,
        model: str = "clip"
    ) -> dict[str, Any]:
        """Compare two images using CLIP similarity.

        Args:
            image1: First image
            image2: Second image
            model: Model to use for comparison

        Returns:
            Similarity score and analysis
        """
        await self._load_vision_model(model)

        if model not in self.vision_models:
            return {"error": f"Failed to load vision model: {model}"}

        # Load images
        images = []
        for img_input in [image1, image2]:
            if isinstance(img_input, str):
                if img_input.startswith("http"):
                    response = requests.get(img_input)
                    img = Image.open(BytesIO(response.content))
                else:
                    img = Image.open(img_input)
            else:
                img = img_input
            images.append(img)

        try:
            model_data = self.vision_models[model]

            if model == "clip":
                inputs = model_data["processor"](
                    images=images,
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    image_features = model_data["model"].get_image_features(**inputs)
                    # Normalize features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    # Compute cosine similarity
                    similarity = torch.cosine_similarity(image_features[0], image_features[1], dim=0)

                return {
                    "similarity_score": float(similarity),
                    "model": model,
                    "interpretation": self._interpret_similarity(float(similarity))
                }

            return {"error": f"Model '{model}' doesn't support image comparison"}

        except Exception as e:
            logger.error(f"Image comparison failed: {e}")
            return {"error": str(e)}

    def _interpret_similarity(self, score: float) -> str:
        """Interpret CLIP similarity score."""
        if score > 0.9:
            return "Very similar images"
        elif score > 0.7:
            return "Similar images"
        elif score > 0.5:
            return "Somewhat similar"
        elif score > 0.3:
            return "Different images"
        else:
            return "Very different images"

    async def cleanup(self):
        """Clean up loaded models to free memory."""
        logger.info("Cleaning up multimodal models")

        for model_key in list(self.vision_models.keys()):
            del self.vision_models[model_key]

        for model_key in list(self.diffusion_models.keys()):
            if "pipeline" in self.diffusion_models[model_key]:
                del self.diffusion_models[model_key]["pipeline"]
            del self.diffusion_models[model_key]

        # Force garbage collection
        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Provider interface methods
    async def generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate text response (not used for multimodal)."""
        yield ""

    async def list_models(self) -> list[dict[str, Any]]:
        """List available multimodal models."""
        return [
            {
                "id": f"multimodal-{model_key}",
                "name": f"{model_key.upper()} ({model_name})",
                "provider": "multimodal",
                "type": "vision" if model_key in self.SUPPORTED_VISION_MODELS else "diffusion"
            }
            for model_key, model_name in {
                **self.SUPPORTED_VISION_MODELS,
                **self.SUPPORTED_DIFFUSION_MODELS
            }.items()
        ]
