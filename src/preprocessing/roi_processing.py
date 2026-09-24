import torch
import numpy as np
from PIL import Image
from typing import Tuple, Optional

def mask_to_bbox(mask: torch.Tensor | np.ndarray | Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """
    Extract bounding box (x_min, y_min, x_max, y_max) from a binary mask.
    Returns None if the mask is empty.
    """
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
        
    # Ensure 2D
    if mask.ndim == 3:
        mask = mask.squeeze()
        
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None
        
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    return int(x_min), int(y_min), int(x_max), int(y_max)

def crop_with_padding(
    image: Image.Image, 
    bbox: Tuple[int, int, int, int], 
    padding: int = 15
) -> Image.Image:
    """
    Crop an image using a bounding box with additional padding.
    """
    x_min, y_min, x_max, y_max = bbox
    width, height = image.size
    
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(width, x_max + padding)
    y_max = min(height, y_max + padding)
    
    return image.crop((x_min, y_min, x_max, y_max))

def apply_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Apply a binary mask to an image (background becomes black).
    """
    img_arr = np.array(image)
    mask_arr = np.array(mask)
    
    # Ensure mask is binary 0/1 (or 0/255)
    if mask_arr.max() == 1:
        mask_arr = mask_arr * 255
        
    if img_arr.ndim == 3 and mask_arr.ndim == 2:
        mask_arr = np.expand_dims(mask_arr, axis=-1)
        
    masked_arr = np.where(mask_arr > 0, img_arr, 0)
    return Image.fromarray(masked_arr.astype(np.uint8))

def extract_roi(image: Image.Image, mask: Image.Image, padding: int = 15) -> Optional[Image.Image]:
    """
    Given an image and its mask, find the bounding box and return the cropped ROI.
    """
    if image.size != mask.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
        
    bbox = mask_to_bbox(mask)
    if bbox is None:
        return None
        
    return crop_with_padding(image, bbox, padding)
