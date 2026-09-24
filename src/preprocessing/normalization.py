"""Intensity normalization routines."""

import torch
import numpy as np


def min_max_normalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Min-Max normalization to scale values between [0, 1].
    
    Formula: (X - min) / (max - min)
    """
    min_val = tensor.min()
    max_val = tensor.max()
    
    if max_val - min_val < 1e-6:
        # Avoid division by zero for constant images
        return torch.zeros_like(tensor)
        
    return (tensor - min_val) / (max_val - min_val)


def z_score_normalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Z-score (standard) normalization.
    
    Formula: (X - mean) / std
    """
    mean_val = tensor.mean()
    std_val = tensor.std()
    
    if std_val < 1e-6:
        return torch.zeros_like(tensor)
        
    return (tensor - mean_val) / std_val


def apply_clahe(image_np: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Requires OpenCV (cv2) to be installed.
    
    Args:
        image_np: 2D numpy array representing a grayscale image.
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalization.
        
    Returns:
        Contrast-enhanced 2D numpy array.
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError("cv2 (opencv-python) is required for CLAHE.") from e

    # Ensure image is 8-bit uint
    if image_np.dtype != np.uint8:
        image_np = (min_max_normalize(torch.from_numpy(image_np)).numpy() * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image_np)
    
    return enhanced
