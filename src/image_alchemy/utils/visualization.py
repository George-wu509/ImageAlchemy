# --- FILENAME: src/image_alchemy/utils/visualization.py ---
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2

def compare_images(
    image_before: Image.Image,
    image_after: Image.Image,
    before_text: str = "Before",
    after_text: str = "After",
    figsize: tuple = (12, 6)
):
    """
    Displays two images side-by-side for comparison.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.imshow(image_before)
    ax1.set_title(before_text)
    ax1.axis('off')

    ax2.imshow(image_after)
    ax2.set_title(after_text)
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

def plot_difference_map(
    image_before: Image.Image,
    image_after: Image.Image,
    figsize: tuple = (18, 6)
):
    """
    Displays the original image, the modified image, and a heatmap of their differences.
    """
    before_np = np.array(image_before.convert('L'))
    after_np = np.array(image_after.resize(image_before.size).convert('L'))

    # Calculate SSIM and difference map
    ssim_score, diff = ssim(before_np, after_np, full=True)
    diff = (diff * 255).astype("uint8")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    ax1.imshow(image_before)
    ax1.set_title("Original")
    ax1.axis('off')

    ax2.imshow(image_after)
    ax2.set_title(f"Modified\nSSIM: {ssim_score:.4f}")
    ax2.axis('off')

    diff_plot = ax3.imshow(diff, cmap='viridis')
    ax3.set_title("Difference Map")
    ax3.axis('off')
    fig.colorbar(diff_plot, ax=ax3)

    plt.tight_layout()
    plt.show()

def plot_histogram_comparison(
    image_before: Image.Image,
    image_after: Image.Image,
    figsize: tuple = (10, 5)
):
    """
    Plots the color histograms of two images to show changes in color distribution.
    """
    before_cv = cv2.cvtColor(np.array(image_before), cv2.COLOR_RGB_BGR)
    after_cv = cv2.cvtColor(np.array(image_after), cv2.COLOR_RGB_BGR)

    colors = ('b', 'g', 'r')
    plt.figure(figsize=figsize)
    
    for i, color in enumerate(colors):
        hist_before = cv2.calcHist([before_cv], [i], None, [256], [0, 256])
        plt.plot(hist_before, color=color, linestyle='--')
        
        hist_after = cv2.calcHist([after_cv], [i], None, [256], [0, 256])
        plt.plot(hist_after, color=color)

    plt.title('Histogram Comparison (Dashed=Before, Solid=After)')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Count')
    plt.xlim([0, 256])
    plt.legend(['Blue (Before)', 'Green (Before)', 'Red (Before)', 'Blue (After)', 'Green (After)', 'Red (After)'])
    plt.grid(True, alpha=0.3)
    plt.show()