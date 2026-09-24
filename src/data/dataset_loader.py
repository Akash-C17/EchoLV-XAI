"""Data loader for the echocardiography dataset."""

from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class EchoSegmentationDataset(Dataset):
    """PyTorch Dataset for Left Ventricle segmentation."""

    def __init__(
        self,
        split_csv: str | Path,
        image_dir: str | Path,
        mask_dir: Optional[str | Path] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize the dataset.

        Args:
            split_csv: Path to the CSV file containing the data split (e.g., train.csv).
            image_dir: Root directory of the images.
            mask_dir: Root directory of the masks. If None, assumes masks are in a 'masks' sibling dir.
            transform: A callable to apply data augmentation/preprocessing.
        """
        self.split_csv = Path(split_csv)
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else self.image_dir.parent / "masks"
        self.transform = transform

        if not self.split_csv.exists():
            raise FileNotFoundError(f"Split CSV not found: {self.split_csv}")
        
        self.data = pd.read_csv(self.split_csv)
        
        # Verify required columns exist
        if "image_path" not in self.data.columns:
            raise ValueError(f"'image_path' column missing in {self.split_csv}")

    def __len__(self) -> int:
        return len(self.data)

    def _get_mask_path(self, img_path: Path) -> Path:
        """Derive the corresponding mask path from the image path."""
        # This assumes the mask has the same name as the image, or with '_mask' appended.
        # You may need to adapt this logic to the specific dataset (e.g. CAMUS).
        stem = img_path.stem
        # e.g., patient0001_2CH_ED -> patient0001_2CH_ED_mask.png
        possible_mask_names = [
            f"{stem}.png",
            f"{stem}_mask.png",
            f"{stem}_gt.png"
        ]
        for name in possible_mask_names:
            mask_path = self.mask_dir / img_path.parent / name
            if mask_path.exists():
                return mask_path
        
        # Fallback to just the stem with .png in the mask dir directly
        return self.mask_dir / f"{stem}.png"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_rel_path = self.data.iloc[idx]["image_path"]
        img_path = self.image_dir / img_rel_path
        mask_path = self._get_mask_path(Path(img_rel_path))

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        # For segmentation, usually convert to grayscale ("L")
        image = Image.open(img_path).convert("L")
        
        if mask_path.exists():
            mask = Image.open(mask_path).convert("L")
        else:
            # If mask doesn't exist, return a black mask of the same size (or handle appropriately)
            mask = Image.new("L", image.size, color=0)

        # Apply spatial transformations consistently if transform is provided
        # The transform function must handle the image and mask as a pair.
        if self.transform is not None:
            # Assumes transform takes a dict or tuple
            image, mask = self.transform(image, mask)
        else:
            from torchvision.transforms import ToTensor
            image = ToTensor()(image)
            mask = ToTensor()(mask)

        return image, mask
