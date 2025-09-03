# Qwen-Ground-SG: A Multi-Modal Library for Scene Graph-Enhanced Visual 


Qwen-Ground-SG is a state-of-the-art Python library that fuses the power of a Vision-Language Model (Qwen2-VL) with the precision of Grounded Segment-Anything (Grounded SAM). It pioneers a novel pipeline that generates dynamic Scene Graphs from visual data to dramatically enhance the model's contextual awareness and reasoning capabilities. This framework provides a unified solution for advanced, high-fidelity image and video analysis.

## Key Features

-   **Advanced Visual Grounding**: Go beyond basic detection. Use natural language prompts to detect, segment, and track multiple objects in both images and videos with high precision.

-   **Dynamic Scene Graph Generation (SGG)**: The core innovation of this library. Automatically construct a structured graph of objects and their semantic relationships within a scene. Generate graphs for the entire scene or for specific, text-queried objects.

-   **Scene Graph-Enhanced VLM**: Leverage the generated scene graph as a rich contextual input for the VLM. This significantly improves performance on complex multi-modal tasks by providing the model with a deeper understanding of how objects relate to one another.

-   **Comprehensive Multi-modal Tasks**:
    -   **Visual Question Answering (VQA)**: Ask complex questions about object interactions, positions, and activities.
    -   **Image & Video Captioning**: Generate detailed, context-aware descriptions of visual scenes.
    -   **Multi-modal Reasoning**: Perform tasks that require understanding the intricate relationships between multiple entities in a scene.

## Core Technology Stack

This project is built on a foundation of cutting-edge models and robust machine learning libraries:

| Component                  | Technology / Library                                                                    |
| -------------------------- | --------------------------------------------------------------------------------------- |
| **Vision-Language Model** | [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL)                                          |
| **Visual Grounding** | [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) & [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) |
| **Core AI Frameworks** | [PyTorch](https://pytorch.org/), [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) |
| **Computer Vision & Utils**| [OpenCV](https://opencv.org/), [Supervision](https://roboflow.github.io/supervision/) |

## Project Structure

The repository is organized to separate the core library logic from examples and tests, promoting modularity and ease of use.

```bash
Qwen-Ground-SG/
├── qwen_ground_sg/              # Core library source code
│   ├── __init__.py
│   ├── main_pipeline.py         # Main orchestrator class
│   ├── vision_language_model.py # Qwen2-VL wrapper
│   ├── grounding_model.py       # Grounded SAM and tracking wrapper
│   ├── scene_graph.py           # Scene Graph Generation logic
│   └── utils.py                 # Image/video processing utilities
├── examples/                    # Usage examples and notebooks
│   ├── image_processing_example.py
│   └── video_processing_example.py
├── pyproject.toml               # Project metadata and dependencies (for pip)
├── environment.yaml             # Conda environment definition
├── setup.py                     # Required for editable installs
└── README.md

```

## Installation

Follow these steps to set up the project and its dependencies. A Conda environment is strongly recommended for managing the complex dependencies.

Prerequisites:
Conda / Miniconda

Step 1: Clone the Repository
Open your terminal and clone the repository to your local machine:

```bash
git clone [https://github.com/George-wu509/Qwen-Ground-SG.git](https://github.com/George-wu509/Qwen-Ground-SG.git)
cd Qwen-Ground-SG
```

Step 2: Create and Activate the Conda Environment
Use the provided environment.yaml file to create a Conda environment with all the necessary packages. This will ensure that all dependencies, including specific PyTorch and CUDA versions, are correctly installed.

```bash
conda env create -f environment.yaml
conda activate qwen-ground-sg
```

Step 3: Install the Library
Install the project in "editable" mode. This allows you to modify the source code and have the changes immediately reflected in your environment.
```bash
pip install -e .
```

Step 4: Download Model Weights
This library requires pre-trained model weights for its components. Create a weights/ directory in the project root and download the necessary files:

1. GroundingDINO: Download the checkpoint file (e.g., groundingdino_swint_ogc.pth) from the official GroundingDINO repository releases.

2. Segment Anything (SAM): Download a checkpoint file (e.g., sam_vit_h_4b8939.pth) from the official SAM repository.

3. Qwen2-VL: The Qwen2-VL model can be downloaded automatically via the Hugging Face Transformers library, so no manual download is needed if you have an internet connection.

Place the downloaded .pth files into the weights/ directory.

## Quick Start

The following example demonstrates how to perform a Scene Graph-Enhanced Visual Question Answering (VQA) task on a single image.

```python
import os
from qwen_ground_sg import VisionPipeline

# 1. Define the configuration with paths to your models
#    Ensure the Qwen model name is correct and checkpoint paths are valid.
#    Create a 'weights' directory in your project root for the .pth files.
config = {
    "qwen_model_path": "Qwen/Qwen2-VL-7B-Instruct",
    "grounding_dino_config_path": "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", # This path might need adjustment
    "grounding_dino_checkpoint_path": "./weights/groundingdino_swint_ogc.pth",
    "sam_checkpoint_path": "./weights/sam_vit_h_4b8939.pth"
}

# Create a directory for outputs
os.makedirs("outputs", exist_ok=True)

# 2. Initialize the main pipeline
#    This will load all the models into memory.
print("Initializing Vision Pipeline...")
pipeline = VisionPipeline(config)
print("Pipeline initialized successfully.")

# 3. Define your inputs
image_path = "path/to/your/image.jpg" # <-- IMPORTANT: Change this to your image path
question = "What is the relationship between the person and the dog?"
output_path = "outputs/detected_objects.jpg"

# 4. Perform Scene Graph-Enhanced VQA
print("\n--- Performing Scene Graph-Enhanced VQA ---")
enhanced_result = pipeline.get_vqa_answer(
    media_path=image_path,
    question=question,
    use_scene_graph=True
)

# 5. Print the results
print(f"\nQuestion: {question}")
print(f"Enhanced Answer: {enhanced_result.get('answer', 'No answer generated.')}")

print("\n--- Scene Graph Analysis ---")
print(enhanced_result.get('scene_graph_analysis', 'No analysis available.'))

# 6. (Optional) Visualize the detected objects for the scene graph
print("\n--- Visualizing Detected Objects ---")
pipeline.visualize_grounding(
    media_path=image_path,
    text_prompt="person, dog", # Use a specific prompt for SGG visualization
    output_path=output_path
)
print(f"Detection visualization saved to {output_path}")
```
