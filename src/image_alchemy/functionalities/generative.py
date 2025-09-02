# --- FILENAME: src/image_alchemy/functionalities/generative.py ---
from PIL import Image
from typing import Union, List
from ..core.model_loader import ModelLoader
from ..core.pipelines import run_sam_segmentation, run_inpaint_pipeline, generative_zoom_step
from .manipulation import ManipulationModule

class GenerativeModule:
    """
    Provides functions for generative tasks like background replacement and zoom.
    """
    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader
        self.manipulation = ManipulationModule(model_loader)

    def generate_background(
        self, 
        image: Image.Image, 
        foreground_mask: Union[Image.Image, List[int]], 
        background_prompt: str
    ) -> Image.Image:
        """
        Replaces the background of an image with a generated scene.
        """
        # Get the foreground mask
        mask = self.manipulation._get_mask(image, foreground_mask)

        # Invert the mask to get the background
        background_mask = Image.fromarray(255 - pil_to_numpy(mask))

        return run_inpaint_pipeline(
            self.model_loader,
            image,
            background_mask,
            background_prompt
        )

    def generative_zoom(
        self, 
        image: Image.Image, 
        prompt: str,
        num_steps: int = 10,
        zoom_factor: float = 1.15
    ) -> List[Image.Image]:
        """
        Creates a sequence of images for a "generative zoom" effect.
        """
        frames = [image]
        current_image = image
        for i in range(num_steps):
            print(f"Generating zoom frame {i+1}/{num_steps}...")
            next_frame = generative_zoom_step(
                self.model_loader,
                current_image,
                prompt,
                zoom_factor=zoom_factor
            )
            frames.append(next_frame)
            current_image = next_frame
        
        return frames

    # Style Transfer would be implemented here, likely using a ControlNet-based
    # img2img approach with a high denoising strength and a descriptive style prompt.