# --- FILENAME: src/image_alchemy/core/model_loader.py ---
import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionInpaintPipeline,
    ControlNetModel,
    AutoencoderKL,
    UniPCMultistepScheduler
)
from segment_anything import sam_model_registry, SamPredictor
import huggingface_hub
import os

class ModelLoader:
    """
    Handles the loading, caching, and management of all AI models.
    This class uses a lazy loading approach: a model is only loaded
    into memory when it's first requested.
    """
    def __init__(self, device: str = 'cuda', cache_dir: str = None):
        self.device = device
        self.cache_dir = cache_dir
        self._models = {}

    def _load_model(self, model_name: str, model_class, **kwargs):
        """Generic model loader with caching."""
        if model_name not in self._models:
            print(f"Loading {model_name}...")
            try:
                self._models[model_name] = model_class.from_pretrained(
                    model_name, cache_dir=self.cache_dir, **kwargs
                ).to(self.device)
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
                return None
        return self._models[model_name]
    
    def get_sd_pipeline(self, model_id="stabilityai/stable-diffusion-2-inpainting"):
        return self._load_model(model_id, StableDiffusionInpaintPipeline, torch_dtype=torch.float16)

    def get_controlnet_pipeline(self, base_model_id="runwayml/stable-diffusion-v1-5", controlnet_model_id="lllyasviel/control_v11p_sd15_inpaint"):
        pipeline_key = f"{base_model_id}+{controlnet_model_id}"
        if pipeline_key not in self._models:
            print(f"Loading ControlNet pipeline: {pipeline_key}")
            controlnet = self._load_model(controlnet_model_id, ControlNetModel, torch_dtype=torch.float16)
            
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                base_model_id,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                cache_dir=self.cache_dir
            ).to(self.device)
            pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
            self._models[pipeline_key] = pipeline
        return self._models[pipeline_key]
        
    def get_sam_predictor(self, model_type="vit_h", checkpoint_name="sam_vit_h_4b8939.pth"):
        predictor_key = f"sam_predictor_{model_type}"
        if predictor_key not in self._models:
            print(f"Loading SAM model: {model_type}")
            checkpoint_url = f"https://dl.fbaipublicfiles.com/segment_anything/{checkpoint_name}"
            
            # Ensure cache directory exists
            sam_cache_dir = os.path.join(self.cache_dir or ".cache", "sam_models")
            os.makedirs(sam_cache_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(sam_cache_dir, checkpoint_name)

            if not os.path.exists(checkpoint_path):
                print(f"Downloading SAM checkpoint to {checkpoint_path}...")
                torch.hub.download_url_to_file(checkpoint_url, checkpoint_path)

            sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(self.device)
            predictor = SamPredictor(sam)
            self._models[predictor_key] = predictor
        return self._models[predictor_key]

    def set_device(self, new_device: str):
        """Move all currently loaded models to a new device."""
        self.device = new_device
        for model_name, model in self._models.items():
            if hasattr(model, 'to'):
                print(f"Moving {model_name} to {new_device}...")
                model.to(new_device)
        # Clear VRAM on the old device if it was a GPU
        if 'cuda' in self.device and torch.cuda.is_available():
            torch.cuda.empty_cache()