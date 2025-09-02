# --- FILENAME: src/image_alchemy/functionalities/manipulation.py ---
from PIL import Image
from typing import Union, List
from ..core.model_loader import ModelLoader
from ..core.pipelines import run_sam_segmentation, run_inpaint_pipeline
from ..utils.image_utils import create_mask_from_box, combine_image_and_mask

class ManipulationModule:
    """
    Provides functions for editing objects and scenes within an image.
    """
    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader

    def _get_mask(self, image: Image.Image, mask_input: Union[Image.Image, List[int]]) -> Image.Image:
        """Helper to get a mask from either a direct mask image or a bounding box."""
        if isinstance(mask_input, Image.Image):
            return mask_input.convert("L")
        elif isinstance(mask_input, list) and len(mask_input) == 4:
             # Use SAM for a precise mask from the bounding box
            return run_sam_segmentation(self.model_loader, image, input_box=mask_input)
        else:
            raise ValueError("mask_input must be a PIL Image or a bounding box list [x1, y1, x2, y2]")

    def inpaint(self, image: Image.Image, mask: Union[Image.Image, List[int]], prompt: str) -> Image.Image:
        """
        Fills in a masked area of an image based on a prompt.
        """
        mask_image = self._get_mask(image, mask)
        return run_inpaint_pipeline(self.model_loader, image, mask_image, prompt)

    def remove_object(self, image: Image.Image, mask: Union[Image.Image, List[int]], prompt: str = "photorealistic background, no objects") -> Image.Image:
        """
        Removes an object from an image, filling the space with a plausible background.
        """
        mask_image = self._get_mask(image, mask)
        # The prompt should describe the background to fill in
        return run_inpaint_pipeline(self.model_loader, image, mask_image, prompt)

    def add_object(self, image: Image.Image, mask: Union[Image.Image, List[int]], prompt: str) -> Image.Image:
        """
        Adds an object to a masked area of an image. Alias for inpaint.
        """
        return self.inpaint(image, mask, prompt)

    def reposition_object(
        self, 
        image: Image.Image, 
        source_mask: Union[Image.Image, List[int]], 
        destination_mask: Union[Image.Image, List[int]],
        object_prompt: str
    ) -> Image.Image:
        """
        Moves an object from a source location to a destination location.
        """
        print("Step 1: Removing object from source location...")
        # First, remove the object from its original location
        removed_image = self.remove_object(image, source_mask)

        print("Step 2: Adding object to destination location...")
        # Then, add the object to the new location
        final_image = self.add_object(removed_image, destination_mask, object_prompt)

        return final_image