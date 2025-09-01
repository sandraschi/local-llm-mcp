"""
Advanced Gradio Demo

This example demonstrates:
1. Custom styling with CSS/JS
2. Video generation with Stable Diffusion
3. Image editing with PIL/OpenCV
4. Interactive components and layouts
"""
import os
import cv2
import time
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import gradio as gr
from pathlib import Path
import tempfile
from typing import List, Tuple, Optional

# Try to import video generation libraries
try:
    from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
    from diffusers import StableVideoDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Diffusers not available. Install with: pip install diffusers")

# Custom CSS for styling
CUSTOM_CSS = """
:root {
    --primary: #4f46e5;
    --primary-dark: #4338ca;
    --secondary: #10b981;
    --dark: #1f2937;
    --light: #f3f4f6;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto;
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4 {
    color: var(--primary) !important;
    font-weight: 700 !important;
}

button {
    background: var(--primary) !important;
    color: white !important;
    border: none !important;
    padding: 10px 20px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
}

button:hover {
    background: var(--primary-dark) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
}

.tab-nav {
    background: white !important;
    border-radius: 12px !important;
    padding: 10px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
}

.output-panel {
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
}
"""

# Image Processing Functions
def apply_filter(image: Image.Image, filter_type: str, strength: float = 1.0) -> Image.Image:
    """Apply various filters to an image."""
    if filter_type == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=strength*5))
    elif filter_type == "sharpen":
        return image.filter(ImageFilter.SHARPEN)
    elif filter_type == "edge_enhance":
        return image.filter(ImageFilter.EDGE_ENHANCE)
    elif filter_type == "emboss":
        return image.filter(ImageFilter.EMBOSS)
    elif filter_type == "grayscale":
        return image.convert('L')
    return image

def adjust_image(image: Image.Image, brightness: float, contrast: float, saturation: float) -> Image.Image:
    """Adjust image properties."""
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(saturation)
    
    return image

def add_text_to_image(image: Image.Image, text: str, position: Tuple[int, int], 
                     font_size: int = 30, color: str = "white") -> Image.Image:
    """Add text to an image."""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Add text with shadow for better visibility
    shadow_position = (position[0]+1, position[1]+1)
    draw.text(shadow_position, text, fill="black", font=font)
    draw.text(position, text, fill=color, font=font)
    return image

# Video Generation Functions
def generate_video(prompt: str, duration: int = 4, fps: int = 8) -> str:
    """Generate a short video from text prompt."""
    if not DIFFUSERS_AVAILABLE:
        return None
    
    # Initialize the pipeline
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipe = pipe.to("cuda")
    
    # Generate image first
    image_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")
    image = image_pipe(prompt).images[0]
    
    # Generate video
    video_frames = pipe(
        image,
        num_frames=duration * fps,
        decode_chunk_size=8,
        motion_bucket_id=180,
        noise_aug_strength=0.1
    ).frames[0]
    
    # Save video
    output_path = os.path.join(tempfile.gettempdir(), f"generated_video_{int(time.time())}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = video_frames[0].size
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in video_frames:
        frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        out.write(frame_cv)
    
    out.release()
    return output_path

# Gradio Interface
def create_advanced_interface():
    """Create an advanced Gradio interface with multiple tabs."""
    with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Default(primary_hue="indigo")) as demo:
        gr.Markdown("""
        # 🎨 Advanced Media Studio
        *Create, edit, and generate images and videos with AI*
        """)
        
        with gr.Tabs() as tabs:
            # Tab 1: Image Editor
            with gr.Tab("🖼️ Image Editor", id="image_editor"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(label="Upload Image", type="pil")
                        
                        with gr.Accordion("Adjustments", open=True):
                            brightness = gr.Slider(0.1, 2.0, value=1.0, label="Brightness")
                            contrast = gr.Slider(0.1, 2.0, value=1.0, label="Contrast")
                            saturation = gr.Slider(0.0, 2.0, value=1.0, label="Saturation")
                        
                        with gr.Accordion("Filters", open=False):
                            filter_type = gr.Dropdown(
                                ["None", "blur", "sharpen", "edge_enhance", "emboss", "grayscale"],
                                label="Filter Type",
                                value="None"
                            )
                            filter_strength = gr.Slider(0.1, 2.0, value=1.0, label="Filter Strength")
                        
                        with gr.Accordion("Text Overlay", open=False):
                            text_input = gr.Textbox(label="Text")
                            text_size = gr.Slider(10, 100, value=30, label="Text Size")
                            text_position_x = gr.Slider(0, 1000, value=50, label="Text X Position")
                            text_position_y = gr.Slider(0, 1000, value=50, label="Text Y Position")
                            text_color = gr.ColorPicker(label="Text Color", value="#ffffff")
                        
                        apply_btn = gr.Button("Apply Changes", variant="primary")
                    
                    with gr.Column(scale=1):
                        image_output = gr.Image(label="Edited Image")
                        download_btn = gr.Button("Download Image")
                
                # Connect components
                inputs = [image_input, brightness, contrast, saturation, filter_type, filter_strength,
                         text_input, text_position_x, text_position_y, text_size, text_color]
                
                def process_image(*args):
                    if args[0] is None:
                        return None
                    
                    img = args[0].copy()
                    
                    # Apply adjustments
                    img = adjust_image(img, args[1], args[2], args[3])
                    
                    # Apply filter if selected
                    if args[4] != "None":
                        img = apply_filter(img, args[4], args[5])
                    
                    # Add text if provided
                    if args[6] and args[6].strip():
                        img = add_text_to_image(
                            img, 
                            args[6], 
                            (int(args[7]), int(args[8])),
                            int(args[9]),
                            args[10]
                        )
                    
                    return img
                
                apply_btn.click(
                    fn=process_image,
                    inputs=inputs,
                    outputs=image_output
                )
                
                # Download functionality
                download_btn.click(
                    fn=lambda img: img.save("edited_image.png") if img else None,
                    inputs=image_output,
                    outputs=None
                )
            
            # Tab 2: Video Generation (if diffusers is available)
            if DIFFUSERS_AVAILABLE:
                with gr.Tab("🎥 Video Generation", id="video_gen"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            video_prompt = gr.Textbox(
                                label="Video Prompt",
                                placeholder="A beautiful sunset over mountains..."
                            )
                            video_duration = gr.Slider(1, 10, value=4, step=1, label="Duration (seconds)")
                            video_fps = gr.Slider(4, 30, value=8, step=1, label="Frames per Second")
                            generate_btn = gr.Button("Generate Video", variant="primary")
                        
                        with gr.Column(scale=1):
                            video_output = gr.Video(label="Generated Video", autoplay=True)
                            status = gr.Textbox(label="Status", interactive=False)
                    
                    def generate_video_wrapper(prompt, duration, fps):
                        if not prompt.strip():
                            return None, "Please enter a prompt"
                        
                        try:
                            video_path = generate_video(prompt, duration, fps)
                            if video_path and os.path.exists(video_path):
                                return video_path, "Video generated successfully!"
                            return None, "Failed to generate video"
                        except Exception as e:
                            return None, f"Error: {str(e)}"
                    
                    generate_btn.click(
                        fn=generate_video_wrapper,
                        inputs=[video_prompt, video_duration, video_fps],
                        outputs=[video_output, status]
                    )
            else:
                with gr.Tab("🎥 Video Generation (Not Available)", id="video_disabled"):
                    gr.Markdown("""
                    ## Video Generation Unavailable
                    
                    To enable video generation, please install the required dependencies:
                    ```bash
                    pip install diffusers transformers accelerate torch torchvision
                    ```
                    
                    Note: You'll also need a CUDA-capable GPU for video generation.
                    """)
            
            # Tab 3: AI Image Generation
            with gr.Tab("🤖 AI Image Generation", id="ai_image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        ai_prompt = gr.Textbox(
                            label="Image Prompt",
                            placeholder="A beautiful landscape with mountains and a lake..."
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="blurry, low quality, distorted..."
                        )
                        
                        with gr.Row():
                            width = gr.Slider(256, 1024, value=512, step=64, label="Width")
                            height = gr.Slider(256, 1024, value=512, step=64, label="Height")
                        
                        num_images = gr.Slider(1, 4, value=1, step=1, label="Number of Images")
                        guidance_scale = gr.Slider(1, 20, value=7.5, step=0.5, label="Guidance Scale")
                        num_steps = gr.Slider(10, 100, value=30, step=1, label="Inference Steps")
                        
                        generate_img_btn = gr.Button("Generate Image", variant="primary")
                    
                    with gr.Column(scale=1):
                        ai_output = gr.Gallery(
                            label="Generated Images",
                            show_label=True,
                            elem_id="gallery",
                            columns=2,
                            height="auto"
                        )
                
                def generate_image(prompt, negative_prompt, width, height, num_images, guidance_scale, num_steps):
                    if not DIFFUSERS_AVAILABLE:
                        return [None], "Diffusers not available. Install with: pip install diffusers"
                    
                    try:
                        pipe = StableDiffusionPipeline.from_pretrained(
                            "runwayml/stable-diffusion-v1-5",
                            torch_dtype=torch.float16
                        ).to("cuda")
                        
                        images = []
                        for _ in range(int(num_images)):
                            image = pipe(
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                width=int(width),
                                height=int(height),
                                num_inference_steps=int(num_steps),
                                guidance_scale=guidance_scale
                            ).images[0]
                            images.append(image)
                        
                        return images, "Generation complete!"
                    except Exception as e:
                        return [None], f"Error: {str(e)}"
                
                generate_img_btn.click(
                    fn=generate_image,
                    inputs=[ai_prompt, negative_prompt, width, height, num_images, guidance_scale, num_steps],
                    outputs=[ai_output, gr.Textbox(visible=False)]
                )
        
        return demo

if __name__ == "__main__":
    # Check for CUDA
    import torch
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Some features may be slow or unavailable.")
    
    # Create and launch the interface
    demo = create_advanced_interface()
    demo.launch(share=True, server_name="0.0.0.0")
