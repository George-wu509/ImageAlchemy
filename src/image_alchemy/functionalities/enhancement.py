# --- FILENAME: src/image_alchemy/functionalities/enhancement.py ---
from PIL import Image
from ..core.model_loader import ModelLoader
from ..utils.image_utils import pil_to_numpy, numpy_to_pil
from controlnet_aux import CannyDetector, HEDdetector
import torch

class EnhancementModule:
    """
    Provides functions for improving image quality (restoration and enhancement).
    """
    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader

    def _run_img2img_enhancement(self, image: Image.Image, prompt: str, control_type: str, denoising_strength: float = 0.5):
        """Helper for enhancement tasks using ControlNet."""
        controlnet_map = {
            "canny": ("lllyasviel/control_v11p_sd15_canny", CannyDetector()),
            "softedge": ("lllyasviel/control_v11p_sd15_softedge", HEDdetector.from_pretrained('lllyasviel/Annotators'))
        }
        
        if control_type not in controlnet_map:
            raise ValueError(f"Unsupported control type: {control_type}")
            
        controlnet_id, preprocessor = controlnet_map[control_type]
        pipeline = self.model_loader.get_controlnet_pipeline(controlnet_model_id=controlnet_id)

        control_image = preprocessor(image)

        result = pipeline(
            prompt,
            image=image,
            control_image=control_image,
            num_inference_steps=25,
            strength=denoising_strength,
            guidance_scale=7.5
        ).images[0]
        
        return result
        
    def denoise(self, image: Image.Image, strength: float = 0.35) -> Image.Image:
        """
        Removes noise from an image.
        Uses a soft-edge ControlNet to preserve structure while regenerating texture.
        """
        prompt = "denoised, clean, sharp, high quality photo, dslr, 8k"
        negative_prompt = "noise, noisy, grainy, blurry, soft"
        return self._run_img2img_enhancement(image, prompt, "softedge", denoising_strength=strength)

    def sharpen(self, image: Image.Image, strength: float = 0.3) -> Image.Image:
        """
        Sharpens a blurry image.
        Uses a Canny edge ControlNet to reinforce edges.
        """
        prompt = "sharp, focused, clear, detailed, high contrast, professional photograph"
        negative_prompt = "blurry, out of focus, soft, hazy"
        return self._run_img2img_enhancement(image, prompt, "canny", denoising_strength=strength)
        
    def deblur(self, image: Image.Image, strength: float = 0.4) -> Image.Image:
        """Alias for sharpen with slightly higher strength."""
        return self.sharpen(image, strength)
        
    def super_resolution(self, image: Image.Image, scale: int = 4, prompt: str = "high resolution, ultra detailed") -> Image.Image:
        """
        Increases image resolution and adds detail.
        Note: This is a placeholder for a true super-resolution model like DiffBIR or SwinIR.
        For now, it resizes and then sharpens to simulate the effect.
        """
        print("Warning: Using a simulated Super-Resolution. For best results, integrate a dedicated SR model.")
        w, h = image.size
        resized_image = image.resize((w * scale, h * scale), Image.LANCZOS)
        
        final_image = self.sharpen(resized_image, strength=0.2)
        return final_image

    def colorize(self, image: Image.Image, prompt: str = "a vivid, realistic color photograph") -> Image.Image:
        """
        Adds color to a black and white image.
        """
        if image.mode == 'RGB':
            image = image.convert('L').convert('RGB') # Ensure it's treated as B&W
            
        return self._run_img2img_enhancement(image, prompt, "canny", denoising_strength=0.9)
        
    def correct_light(self, image: Image.Image, prompt: str = "good lighting, well-lit, balanced light, studio lighting") -> Image.Image:
        """
        Corrects poor lighting in an image.
        """
        return self._run_img2img_enhancement(image, prompt, "softedge", denoising_strength=0.45)
    
    # ... Other enhancement functions like Dehaze, Fix White Balance, Apply to HDR
    # would be implemented here, potentially using specialized LoRAs or models
    # which would be loaded via the model_loader.