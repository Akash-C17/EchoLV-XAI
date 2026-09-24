import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

def generate_overlay(image_arr: np.ndarray, heatmap: np.ndarray, alpha=0.5):
    """
    Overlay a heatmap onto an image.
    Args:
        image_arr: Base image in RGB (H, W, 3) or grayscale (H, W). uint8.
        heatmap: Heatmap values in [0, 1] (H, W).
    """
    if image_arr.ndim == 2:
        image_arr = cv2.cvtColor(image_arr, cv2.COLOR_GRAY2RGB)
        
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(image_arr, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

def plot_gradcam(original_img, mask, heatmap, overlay, target_class, output_path):
    """
    Plot and save the Grad-CAM visualization.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Original Image')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('LV Segmentation')
    
    axes[2].imshow(heatmap, cmap='jet')
    axes[2].set_title(f'Grad-CAM (Class {target_class})')
    
    axes[3].imshow(overlay)
    axes[3].set_title('Grad-CAM Overlay')
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
