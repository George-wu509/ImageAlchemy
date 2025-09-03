# ImageAlchemy: Prompt-Driven Image Enhancement and Generative Super Resolution:

**Vision-Craft** is a powerful Python library designed to simplify the application of generative AI in image processing. This project abstracts the complexity of state-of-the-art models like **Stable Diffusion**, **ControlNet**, and the **Segment Anything Model (SAM)** into an intuitive, high-level API. Whether you want to restore old photos, edit images with text commands, or create entirely new visual compositions, Vision-Craft enables you to achieve professional-grade results with just a few lines of code.

## Key Features

The library encapsulates complex generative pipelines into a suite of powerful and easy-to-use functions:

**Enhancement & Restoration**
-   **Super Resolution**: Generatively upscales images, enhancing resolution and intelligently adding fine details.

-   **Denoise / Deblur**: Removes noise and motion blur, resulting in a sharper, cleaner image.

-   **Colorize**: Automatically adds natural and realistic color to black-and-white photos.

-   **Dehaze**: Eliminates fog or haze from landscape shots, restoring the scene's true colors.

-   **Correct Light**: Adjusts the lighting and mood of an image based on a text prompt.

-   **Apply HDR**: Generates a High Dynamic Range (HDR) effect, enhancing detail in shadows and highlights.

**Object & Scene Manipulation**
-   **Remove Object**: Seamlessly removes any object from an image, powered by SAM's precise masking.

-   **Add / Reposition Object**: Adds a new object to a specified location or moves an existing one.

-   **Inpainting**: Intelligently fills in missing or corrupted areas of an image.

-   **Generate Background**: Replaces an image's background with a new scene generated from a prompt.

**Creative & Generative**
-   **Style Transfer**: Transforms an image into a different artistic style.

-   **Generative Zoom**: Creates "infinite zoom" video effects through iterative outpainting.

-   **Prompt-driven Edits**: Performs global or local modifications to an image based on text instructions.


## Core Technology Stack
Vision-Craft is powered by industry-leading technologies and libraries:

-   **Deep Learning Framework**: PyTorch

-   **Core Generative Models**: Stable Diffusion (v1.5, SDXL), ControlNet

-   **Precision Segmentation Model**: Segment Anything Model (SAM)

-   **Model & Pipeline Management**: Hugging Face (diffusers, transformers, accelerate)

-   **Image Processing Utilities**: OpenCV, Pillow (PIL)



## Project Structure
This project follows modern Python packaging best practices, adopting the src layout for clarity and maintainability.

```bash
vision-craft/
├── .github/              # CI/CD Workflows (GitHub Actions)
├── examples/             # Example scripts and Colab notebooks
├── src/
│   └── vision_craft/     # Library source code
│       ├── __init__.py
│       ├── core/         # Model loading, core pipelines
│       ├── functionalities/ # Implementations of key features
│       ├── utils/        # Visualization and image utilities
│       └── craft.py      # Main user-facing class
├── .gitignore
├── environment.yml       # Conda environment definition
├── LICENSE
├── pyproject.toml        # Python project definition and dependencies
└── README.md
```

## Installation

We strongly recommend using Conda to manage the environment, especially to ensure CUDA version compatibility.

1. Clone the Repository
```bash
git clone https://github.com/[your-username]/vision-craft.git
cd vision-craft
```

2. Create and Activate the Conda Environment
```bash
conda env create -f environment.yml
conda activate vision-craft
```

3. Install the Library in Editable Mode
This allows you to make changes to the source code and have them immediately reflected.

```bash
pip install -e .
```

## Quick Start

Experience the power of Vision-Craft with this simple example. The following code shows how to upscale an image and then remove an object from it.

```python
from PIL import Image
from vision_craft import VisionCraft
from vision_craft.utils.visualization import compare_images

# 1. Initialize the VisionCraft Engine
# Models will be automatically downloaded on the first run
engine = VisionCraft(device='cuda')

# 2. Load Your Image
# Assume you have an image named "my_photo.jpg" in the same directory
try:
    input_image = Image.open("my_photo.jpg")
except FileNotFoundError:
    print("Error: Please place your image 'my_photo.jpg' in this directory.")
    # As a fallback, let's download a sample image from the web
    import requests
    from io import BytesIO
    url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat-dog.png'
    response = requests.get(url)
    input_image = Image.open(BytesIO(response.content)).convert("RGB")


# 3. Perform Generative Super Resolution
print("Performing 2x Super Resolution...")
sr_image = engine.enhancement.super_resolution(
    image=input_image,
    scale=2,
    prompt="a high-resolution, ultra-detailed photograph of a cat and a dog"
)

# 4. Remove an Object from the Upscaled Image (e.g., the dog)
# We only need to provide an approximate bounding box: [x1, y1, x2, y2]
print("Removing an object...")
object_bounding_box = [350, 20, 650, 350]

final_image = engine.manipulation.remove_object(
    image=sr_image,
    mask=object_bounding_box,
    prompt="a cat sitting on a rug, high quality, photorealistic"
)

# 5. Visualize and Save the Result
print("Processing complete!")
compare_images(input_image, final_image, before_text="Original Image", after_text="Final Result")
final_image.save("vision_craft_output.png")
print("Final result saved as 'vision_craft_output.png'")
```