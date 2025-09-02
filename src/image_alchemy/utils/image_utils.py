# --- FILENAME: src/image_alchemy/utils/image_utils.py ---
import numpy as np
from PIL import Image
from typing import List

def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a NumPy array."""
    return np.array(image.convert("RGB"))

def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """Convert a NumPy array to a PIL Image."""
    return Image.fromarray(array.astype(np.uint8))

def create_mask_from_box(image_size: tuple, box: List[int]) -> Image.Image:
    """Creates a binary mask image from a bounding box."""
    width, height = image_size
    mask = Image.new('L', (width, height), 0)
    mask_np = np.array(mask)
    x1, y1, x2, y2 = box
    mask_np[y1:y2, x1:x2] = 255
    return Image.fromarray(mask_np)

def combine_image_and_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Overlays a semi-transparent mask on an image for visualization."""
    image_rgba = image.convert("RGBA")
    mask_rgba = mask.convert("L").point(lambda p: p > 128 and 128).convert("RGBA")
    
    # Create a red overlay from the mask
    red_overlay = Image.new("RGBA", image.size, (255, 0, 0, 0))
    red_overlay.paste((255,0,0,128), mask=mask.convert("L"))

    return Image.alpha_composite(image_rgba, red_overlay)