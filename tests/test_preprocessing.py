import torch
from PIL import Image
import numpy as np

from src.preprocessing.image_preprocessing import preprocess_image_and_mask
from src.preprocessing.normalization import min_max_normalize, z_score_normalize
from src.data.transforms import ComposeTransforms, RandomHorizontalFlipDual

def test_preprocess_image_and_mask():
    # Create fake RGB image
    img = Image.new("RGB", (300, 400), color=(100, 100, 100))
    mask = Image.new("L", (300, 400), color=1)
    
    p_img, p_mask = preprocess_image_and_mask(img, mask, target_size=(256, 256))
    
    assert p_img.mode == "L", "Image should be converted to Grayscale"
    assert p_mask.mode == "L", "Mask should be Grayscale"
    assert p_img.size == (256, 256), "Image should be resized"
    assert p_mask.size == (256, 256), "Mask should be resized"

def test_min_max_normalize():
    tensor = torch.tensor([[-1.0, 0.0], [1.0, 3.0]])
    norm = min_max_normalize(tensor)
    
    assert torch.isclose(norm.min(), torch.tensor(0.0)), "Min should be 0"
    assert torch.isclose(norm.max(), torch.tensor(1.0)), "Max should be 1"

def test_z_score_normalize():
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    norm = z_score_normalize(tensor)
    
    assert torch.isclose(norm.mean(), torch.tensor(0.0), atol=1e-6), "Mean should be 0"
    assert torch.isclose(norm.std(), torch.tensor(1.0), atol=1e-6) or torch.isclose(norm.std(unbiased=False), torch.tensor(1.0), atol=1e-6), "Std should be roughly 1"

def test_random_horizontal_flip_dual():
    img_tensor = torch.arange(4).view(1, 2, 2).float()
    mask_tensor = torch.arange(4).view(1, 2, 2).float()
    
    transform = RandomHorizontalFlipDual(p=1.0)
    flip_img, flip_mask = transform(img_tensor, mask_tensor)
    
    # Check spatial consistency
    assert torch.equal(flip_img, flip_mask), "Image and mask should undergo the exact same transformation"
    assert flip_img[0, 0, 0].item() == 1, "Should be horizontally flipped"
