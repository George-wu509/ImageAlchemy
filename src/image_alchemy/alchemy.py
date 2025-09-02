# --- FILENAME: src/image_alchemy/alchemy.py ---
import torch
from .core.model_loader import ModelLoader
from .functionalities.enhancement import EnhancementModule
from .functionalities.manipulation import ManipulationModule
from .functionalities.generative import GenerativeModule

class ImageAlchemy:
    """
    The main user-facing class for the ImageAlchemy library.
    This class initializes all necessary models and provides access to
    different image processing modules.
    """
    def __init__(self, device: str = None, cache_dir: str = None):
        """
        Initializes the ImageAlchemy engine.

        Args:
            device (str, optional): The device to run the models on ('cuda', 'cpu'). 
                                    If None, automatically detects GPU. Defaults to None.
            cache_dir (str, optional): The directory to cache downloaded models. 
                                       Defaults to Hugging Face's default cache.
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Initializing ImageAlchemy on device: {self.device}")

        self.model_loader = ModelLoader(device=self.device, cache_dir=cache_dir)

        # Initialize functional modules
        self.enhancement = EnhancementModule(self.model_loader)
        self.manipulation = ManipulationModule(self.model_loader)
        self.generative = GenerativeModule(self.model_loader)

        print("ImageAlchemy engine initialized successfully.")

    def set_device(self, device: str):
        """
        Changes the device for all loaded models.
        Note: This can be slow as it involves moving models in memory.
        """
        print(f"Setting device to: {device}")
        self.device = device
        self.model_loader.set_device(device)