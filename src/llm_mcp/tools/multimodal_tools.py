"""Multimodal tools for the LLM MCP server.

This module provides tools for working with multimodal models, including:
- Image analysis and description
- Image generation from text prompts
- Image similarity search
- Basic image processing
"""
import base64
import io
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from PIL import Image
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from sentence_transformers import SentenceTransformer, util
    from transformers import pipeline
    import torch
    HAS_MM_DEPS = True
except ImportError:
    HAS_MM_DEPS = False
    logger.warning("Multimodal dependencies not installed. Install with 'pip install transformers sentence-transformers Pillow'")

# Type aliases
ImageInput = Union[str, bytes, Image.Image]

class ImageAnalysisResult(BaseModel):
    """Result of image analysis."""
    description: str
    tags: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class GeneratedImage(BaseModel):
    """Generated image result."""
    image_data: bytes
    mime_type: str = "image/png"
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MultimodalTools:
    """Tools for working with multimodal models."""
    
    def __init__(self):
        self.image_model = None
        self.text_to_image_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_image(self, image_input: ImageInput) -> Image.Image:
        """Load an image from various input formats."""
        if isinstance(image_input, Image.Image):
            return image_input
        elif isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, str):
            if image_input.startswith('http'):
                import requests
                response = requests.get(image_input, stream=True)
                response.raise_for_status()
                return Image.open(io.BytesIO(response.content))
            else:
                return Image.open(image_input)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
    
    async def analyze_image(
        self, 
        image: ImageInput,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        **kwargs
    ) -> ImageAnalysisResult:
        """Analyze an image and generate a description.
        
        Args:
            image: Input image (path, URL, bytes, or PIL Image)
            model_name: Name of the image analysis model to use
            **kwargs: Additional arguments for the model
            
        Returns:
            ImageAnalysisResult with description and metadata
        """
        if not HAS_MM_DEPS:
            raise ImportError("Multimodal dependencies not installed. Install with 'pip install transformers sentence-transformers Pillow'")
            
        # Load model if not already loaded
        if self.image_model is None or self.image_model.name_or_path != model_name:
            self.image_model = pipeline(
                "image-to-text", 
                model=model_name,
                device=self.device
            )
        
        # Process image
        pil_image = self.load_image(image)
        result = self.image_model(pil_image, **kwargs)
        
        # Format result
        if isinstance(result, list) and len(result) > 0:
            description = result[0].get('generated_text', 'No description generated')
        else:
            description = str(result)
            
        # Generate some basic tags (this could be enhanced with a tagger model)
        tags = self._generate_tags(description)
        
        return ImageAnalysisResult(
            description=description,
            tags=tags,
            metadata={
                "model": model_name,
                "device": self.device
            }
        )
    
    async def generate_image(
        self,
        prompt: str,
        model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
        **kwargs
    ) -> GeneratedImage:
        """Generate an image from a text prompt.
        
        Args:
            prompt: Text prompt for image generation
            model_name: Name of the text-to-image model to use
            **kwargs: Additional arguments for the model
            
        Returns:
            GeneratedImage containing the image data and metadata
        """
        if not HAS_MM_DEPS:
            raise ImportError("Multimodal dependencies not installed. Install with 'pip install diffusers transformers torch")
            
        from diffusers import StableDiffusionPipeline
        import torch
        
        # Load model if not already loaded
        if self.text_to_image_model is None or self.text_to_image_model.name_or_path != model_name:
            self.text_to_image_model = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True
            ).to(self.device)
        
        # Generate image
        with torch.inference_mode():
            result = self.text_to_image_model(prompt, **kwargs)
            
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        result.images[0].save(img_byte_arr, format='PNG')
        
        return GeneratedImage(
            image_data=img_byte_arr.getvalue(),
            mime_type="image/png",
            metadata={
                "model": model_name,
                "prompt": prompt,
                **kwargs
            }
        )
    
    async def image_similarity(
        self,
        image1: ImageInput,
        image2: ImageInput,
        model_name: str = "clip-ViT-B-32"
    ) -> float:
        """Calculate the similarity between two images.
        
        Returns a similarity score between 0 and 1, where 1 is identical.
        """
        if not HAS_MM_DEPS:
            raise ImportError("Multimodal dependencies not installed. Install with 'pip install sentence-transformers")
            
        # Load model
        model = SentenceTransformer(f'clip-{model_name}')
        
        # Process images
        img1 = self.load_image(image1)
        img2 = self.load_image(image2)
        
        # Get embeddings
        embedding1 = model.encode([img1], convert_to_tensor=True)
        embedding2 = model.encode([img2], convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = util.pytorch_cos_sim(embedding1, embedding2)[0][0].item()
        return float(similarity)
    
    def _generate_tags(self, text: str) -> List[str]:
        """Generate tags from text (placeholder implementation)."""
        # This is a simple implementation - could be replaced with a proper tagger model
        import re
        from collections import Counter
        
        # Remove special characters and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Filter out common words and get top 5
        common_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'is', 'are', 'was', 'were'}
        tags = [word for word, _ in word_counts.most_common(10) 
                if word not in common_words and len(word) > 2]
                
        return tags[:5]  # Return top 5 tags

# Create a global instance
multimodal_tools = MultimodalTools()

def register_multimodal_tools(mcp):
    """Register multimodal tools with the MCP server."""
    if not HAS_MM_DEPS:
        logger.warning("Not registering multimodal tools - dependencies not installed")
        return
    
    @mcp.tool()
    async def analyze_image(
        image: str,
        model_name: str = "Salesforce/blip2-opt-2.7b"
    ) -> Dict[str, Any]:
        """Analyze an image and generate a description.
        
        Args:
            image: Path to image, URL, or base64-encoded image
            model_name: Model to use for analysis
            
        Returns:
            Dictionary with analysis results
        """
        result = await multimodal_tools.analyze_image(image, model_name=model_name)
        return result.dict()
    
    @mcp.tool()
    async def generate_image(
        prompt: str,
        model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5
    ) -> Dict[str, Any]:
        """Generate an image from a text prompt.
        
        Args:
            prompt: Text prompt for image generation
            model_name: Model to use for generation
            negative_prompt: Negative prompt to avoid certain features
            width: Width of the generated image
            height: Height of the generated image
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for the model
            
        Returns:
            Dictionary with base64-encoded image and metadata
        """
        result = await multimodal_tools.generate_image(
            prompt=prompt,
            model_name=model_name,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        
        # Convert image to base64 for JSON serialization
        image_b64 = base64.b64encode(result.image_data).decode('utf-8')
        
        return {
            "image": f"data:{result.mime_type};base64,{image_b64}",
            "metadata": result.metadata
        }
    
    @mcp.tool()
    async def compare_images(
        image1: str,
        image2: str,
        model_name: str = "clip-ViT-B-32"
    ) -> Dict[str, float]:
        """Compare two images and return a similarity score.
        
        Returns a score between 0 and 1, where 1 is identical.
        """
        similarity = await multimodal_tools.image_similarity(image1, image2, model_name=model_name)
        return {"similarity": similarity}
