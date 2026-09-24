import pytest
import numpy as np
from PIL import Image

from src.preprocessing.roi_processing import mask_to_bbox, crop_with_padding, apply_mask, extract_roi

def test_mask_to_bbox():
    mask = np.zeros((100, 100))
    mask[20:40, 30:60] = 1
    
    bbox = mask_to_bbox(mask)
    assert bbox == (30, 20, 59, 39), "Bounding box should tightly wrap the mask"
    
def test_mask_to_bbox_empty():
    mask = np.zeros((100, 100))
    bbox = mask_to_bbox(mask)
    assert bbox is None, "Empty mask should return None bbox"

def test_crop_with_padding():
    img = Image.new("L", (100, 100), color=255)
    bbox = (20, 20, 40, 40)
    
    cropped = crop_with_padding(img, bbox, padding=5)
    assert cropped.size == (30, 30), "Cropped size should include padding on all sides"
    
    # Test boundary padding (should not exceed image size)
    bbox_edge = (0, 0, 10, 10)
    cropped_edge = crop_with_padding(img, bbox_edge, padding=10)
    assert cropped_edge.size == (20, 20), "Padding shouldn't go outside image boundaries"

def test_apply_mask():
    img = np.ones((50, 50)) * 255
    mask = np.zeros((50, 50))
    mask[10:20, 10:20] = 1
    
    img_pil = Image.fromarray(img.astype(np.uint8))
    mask_pil = Image.fromarray(mask.astype(np.uint8))
    
    masked = apply_mask(img_pil, mask_pil)
    masked_arr = np.array(masked)
    
    assert masked_arr[15, 15] == 255, "Inside mask should be preserved"
    assert masked_arr[5, 5] == 0, "Outside mask should be black"
