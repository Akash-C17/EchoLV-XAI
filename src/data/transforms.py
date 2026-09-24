"""Custom data augmentations maintaining spatial consistency for image and mask."""

import random
from typing import Tuple
import torch
import torchvision.transforms.functional as F
from PIL import Image

class ComposeTransforms:
    """Composes several dual-transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class ToTensorDual:
    """Convert PIL Images to Tensors."""
    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        return F.to_tensor(image), F.to_tensor(mask)


class RandomHorizontalFlipDual:
    """Horizontally flip the given image and mask randomly with a given probability."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.p:
            return F.hflip(image), F.hflip(mask)
        return image, mask


class RandomRotationDual:
    """Rotate the image and mask by a random angle."""
    def __init__(self, degrees: Tuple[float, float]):
        self.degrees = degrees

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        angle = random.uniform(self.degrees[0], self.degrees[1])
        # Mask requires NEAREST interpolation to maintain binary classes
        return (
            F.rotate(image, angle, interpolation=F.InterpolationMode.BILINEAR),
            F.rotate(mask, angle, interpolation=F.InterpolationMode.NEAREST)
        )
