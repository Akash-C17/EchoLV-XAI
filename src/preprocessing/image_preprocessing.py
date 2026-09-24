"""Basic image preprocessing logic (Grayscale, Resize)."""

from typing import Tuple
from PIL import Image

def preprocess_image_and_mask(
    image: Image.Image,
    mask: Image.Image,
    target_size: Tuple[int, int] = (256, 256)
) -> Tuple[Image.Image, Image.Image]:
    """
    Apply standard preprocessing to an image and its corresponding mask.
    
    - Converts image and mask to Grayscale.
    - Resizes image using bilinear or bicubic interpolation.
    - Resizes mask using nearest-neighbor interpolation to preserve binary labels.
    
    Args:
        image: The input PIL Image.
        mask: The corresponding label PIL Image mask.
        target_size: Tuple of (width, height) to resize to.
        
    Returns:
        Tuple of (preprocessed_image, preprocessed_mask).
    """
    # 1. Convert to grayscale
    if image.mode != "L":
        image = image.convert("L")
    if mask.mode != "L":
        mask = mask.convert("L")
        
    # 2. Resize
    # Image can be resized with standard methods (e.g. BILINEAR / BICUBIC)
    image = image.resize(target_size, resample=Image.Resampling.BILINEAR)
    
    # Mask MUST be resized with NEAREST to prevent interpolation artifacts on class labels
    mask = mask.resize(target_size, resample=Image.Resampling.NEAREST)
    
    return image, mask
