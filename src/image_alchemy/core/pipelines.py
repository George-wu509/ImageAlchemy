# --- FILENAME: src/image_alchemy/core/pipelines.py ---
import numpy as np
import torch
from PIL import Image
from typing import List, Union

from .model_loader import ModelLoader
from ..utils.image_utils import pil_to_numpy, numpy_to_pil, create_mask_from_box

def run_sam_segmentation(
    model_loader: ModelLoader, 
    image: Image.Image, 
    input_box: List[int] = None, 
    input_points: List = None
) -> Image.Image:
    """
    Runs SAM to get a mask for a specified object.

    Args:
        model_loader (ModelLoader): The model loader instance.
        image (Image.Image): The input image.
        input_box (List[int], optional): Bounding box [x1, y1, x2, y2]. Defaults to None.
        input_points (List, optional): Points [[x, y, label], ...]. Defaults to None.

    Returns:
        Image.Image: The generated binary mask as a PIL Image.
    """
    predictor = model_loader.get_sam_predictor()
    image_np = pil_to_numpy(image)
    predictor.set_image(image_np)

    if input_box is None and input_points is None:
        raise ValueError("Either input_box or input_points must be provided for segmentation.")

    box_np = np.array(input_box) if input_box else None
    
    masks, scores, _ = predictor.predict(
        box=box_np,
        multimask_output=False # Get the most likely mask
    )
    
    # Convert single mask (H, W) to (H, W, 1) and then to PIL
    mask = masks[0] 
    mask_pil = Image.fromarray(mask)
    return mask_pil

def run_inpaint_pipeline(
    model_loader: ModelLoader,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str = "low quality, blurry, ugly, deformed",
    strength: float = 1.0,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30
) -> Image.Image:
    """
    Runs a generic inpainting pipeline.

    Args:
        model_loader (ModelLoader): The model loader instance.
        image (Image.Image): The source image.
        mask (Image.Image): The mask for the inpainting area (white is inpaint).
        prompt (str): The prompt describing what to inpaint.
        negative_prompt (str, optional): The negative prompt. Defaults to "low quality...".
        strength (float, optional): Denoising strength. Defaults to 1.0.
        guidance_scale (float, optional): CFG scale. Defaults to 7.5.
        num_inference_steps (int, optional): Number of diffusion steps. Defaults to 30.

    Returns:
        Image.Image: The inpainted image.
    """
    # Using a standard SD inpainting model for robustness
    pipeline = model_loader.get_sd_pipeline()
    
    # Ensure image and mask have the same size and mode
    image = image.convert("RGB").resize(mask.size)

    result_image = pipeline(
        prompt=prompt,
        image=image,
        mask_image=mask,
        negative_prompt=negative_prompt,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).images[0]

    return result_image

def generative_zoom_step(
    model_loader: ModelLoader,
    image: Image.Image,
    prompt: str,
    zoom_factor: float = 1.25,
    steps: int = 25
) -> Image.Image:
    """
    Performs one step of a generative zoom (outpainting).
    This is a simplified example. A full implementation would involve video generation.

    Args:
        model_loader (ModelLoader): The model loader instance.
        image (Image.Image): The current frame/image.
        prompt (str): The prompt for the scene.
        zoom_factor (float, optional): How much to zoom out. Defaults to 1.25.
        steps (int, optional): Inference steps. Defaults to 25.

    Returns:
        Image.Image: The next frame in the zoom sequence.
    """
    w, h = image.size
    new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
    
    # Create a larger canvas and paste the current image in the center
    canvas = Image.new("RGB", (new_w, new_h))
    paste_x = (new_w - w) // 2
    paste_y = (new_h - h) // 2
    canvas.paste(image, (paste_x, paste_y))
    
    # Create a mask for the area to be outpainted
    mask = Image.new("L", (new_w, new_h), 255)
    mask_paste = Image.new("L", (w,h), 0)
    mask.paste(mask_paste, (paste_x, paste_y))
    
    # Use the inpainting pipeline to "outpaint" the new areas
    outpainted_image = run_inpaint_pipeline(
        model_loader=model_loader,
        image=canvas,
        mask=mask,
        prompt=prompt,
        num_inference_steps=steps
    )
    
    return outpainted_image