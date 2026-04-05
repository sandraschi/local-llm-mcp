"""LLM Multimodal portmanteau tool for Local LLM MCP server.

This tool consolidates all image analysis, generation, and comparison operations
into a single interface following the portmanteau pattern.
"""

import logging
from typing import Dict, Any, Optional

from llm_mcp.tools.multimodal_tools import analyze_image_impl, generate_image_impl, image_similarity_impl
from llm_mcp.utils.logging import get_logger

logger = get_logger(__name__)

# Import FastMCP components
try:
    from fastmcp import FastMCP
    from fastmcp.tools import Tool
    FASTMCP_AVAILABLE = True
except ImportError:
    logger.error("FastMCP not available - portmanteau tools require FastMCP >= 2.12.0")
    FASTMCP_AVAILABLE = False

async def llm_multimodal(
    operation: str,
    # Image operations
    image: Optional[str] = None,
    image1: Optional[str] = None,
    image2: Optional[str] = None,
    # Analysis parameters
    model_name: Optional[str] = None,
    # Generation parameters
    prompt: Optional[str] = None,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
) -> Dict[str, Any]:
    """Comprehensive multimodal (image) tool for Local LLM MCP server.

    PORTMANTEAU PATTERN RATIONALE:
    Instead of creating 3 separate multimodal tools (one per operation), this tool consolidates
    related operations into a single interface. Prevents tool explosion (3 tools → 1 tool) while maintaining
    full functionality and improving discoverability. Follows FastMCP 2.13+ best practices.

    SUPPORTED OPERATIONS:
    - analyze_image: Analyze and describe image content (requires image)
    - generate_image: Generate image from text prompt (requires prompt)
    - compare_images: Compare two images for similarity (requires image1, image2)

    Args:
        operation: Operation to perform (analyze_image, generate_image, compare_images)
        image: Image path/URL/base64 for analyze_image operation
        image1: First image for compare_images operation
        image2: Second image for compare_images operation
        model_name: Model to use (optional, defaults vary by operation)
        prompt: Text prompt for generate_image operation
        negative_prompt: Negative prompt for generate_image (default: "")
        width: Image width for generate_image (default: 512)
        height: Image height for generate_image (default: 512)
        num_inference_steps: Inference steps for generate_image (default: 50)
        guidance_scale: Guidance scale for generate_image (default: 7.5)

    Returns:
        Operation-specific result dictionary with image analysis/generation results
    """
    try:
        if operation == "analyze_image":
            if not image:
                return {"error": "image required for analyze_image operation"}

            # Support multiple SOTA vision models
            supported_models = {
                "llava": "llava-hf/llava-1.5-7b-hf",  # Vision-Language (VQA)
                "blip2": "Salesforce/blip2-opt-2.7b",  # Image Captioning (SOTA)
                "blip": "Salesforce/blip-image-captioning-base",  # Fast captioning
                "clip": "openai/clip-vit-base-patch32",  # Classification/embedding
                "git": "microsoft/git-base",  # Generative Image-to-Text
            }

            model_key = model_name or "blip2"  # Default to BLIP-2 (most capable)
            hf_model_name = supported_models.get(model_key, model_key)

            return await analyze_image_impl(
                image=image,
                model_name=hf_model_name
            )

        elif operation == "generate_image":
            if not prompt:
                return {"error": "prompt required for generate_image operation"}

            # Support multiple SOTA diffusion models
            supported_models = {
                "flux": "blackforestlabs/FLUX.1-dev",
                "sdxl-turbo": "stabilityai/sdxl-turbo",
                "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
                "sd": "runwayml/stable-diffusion-v1-5",
                "realistic-vision": "SG161222/Realistic_Vision_V5.1_noVAE",
                "anything-v5": "andite/anything-v5-better-vae",
            }

            model_key = model_name or "flux"  # Default to FLUX (SOTA)
            hf_model_name = supported_models.get(model_key, model_key)

            # Use optimized parameters for different models
            if model_key == "flux":
                # FLUX is the most advanced - use higher resolution
                width = min(width, 2048)  # FLUX supports up to 2048
                height = min(height, 2048)
                num_inference_steps = min(num_inference_steps, 28)  # FLUX optimal range
                guidance_scale = 0.0  # FLUX doesn't use guidance

            elif model_key == "sdxl-turbo":
                # Turbo is fast but lower quality - single step
                num_inference_steps = 1
                guidance_scale = 0.0

            elif model_key == "sdxl":
                # SDXL optimal settings
                width = min(width, 1024)
                height = min(height, 1024)

            return await generate_image_impl(
                prompt=prompt,
                model_name=hf_model_name,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )

        elif operation == "compare_images":
            if not image1 or not image2:
                return {"error": "image1 and image2 required for compare_images operation"}
            return await image_similarity_impl(
                image1=image1,
                image2=image2,
                model_name=model_name or "clip-ViT-B-32"
            )

        else:
            return {
                "error": f"Unknown operation: {operation}",
                "available_operations": ["analyze_image", "generate_image", "compare_images"]
            }

    except Exception as e:
        logger.error(f"Error in llm_multimodal operation {operation}: {e}", exc_info=True)
        return {"error": f"Operation failed: {str(e)}", "operation": operation}

def register_llm_multimodal_tools(mcp):
    """Register the LLM Multimodal portmanteau tool with the MCP server."""
    if not FASTMCP_AVAILABLE:
        logger.error("Cannot register LLM Multimodal tools - FastMCP not available")
        return mcp

    @mcp.tool()
    async def llm_multimodal_tool(
        operation: str,
        image: Optional[str] = None,
        image1: Optional[str] = None,
        image2: Optional[str] = None,
        model_name: Optional[str] = None,
        prompt: Optional[str] = None,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> Dict[str, Any]:
        """LLM Multimodal Portmanteau Tool - Consolidated image operations.

        This tool consolidates all image analysis, generation, and comparison operations
        into a single interface, reducing the number of MCP tools while maintaining full functionality.

        Use the 'operation' parameter to specify what you want to do:
        - analyze_image: Analyze and describe image content
        - generate_image: Generate images from text prompts
        - compare_images: Compare similarity between two images
        """
        return await llm_multimodal(
            operation=operation,
            image=image,
            image1=image1,
            image2=image2,
            model_name=model_name,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

    logger.info("Registered LLM Multimodal portmanteau tool")
    return mcp
