from pathlib import Path
from typing import Callable, Optional, Tuple
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class EchoClassificationDataset(Dataset):
    def __init__(
        self,
        split_csv: str | Path,
        image_dir: str | Path,
        transform: Optional[Callable] = None,
        mode: str = "full",  # "full", "crop", "masked"
        mask_dir: Optional[str | Path] = None
    ):
        """
        Args:
            split_csv: Path to CSV containing 'image_path' and 'label' (if available).
            image_dir: Root directory of images.
            transform: PyTorch transforms.
            mode: Determines what image to load (full, crop, masked).
            mask_dir: Required if mode is 'crop' or 'masked'.
        """
        self.split_csv = Path(split_csv)
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.transform = transform
        self.mode = mode

        if not self.split_csv.exists():
            raise FileNotFoundError(f"Split CSV not found: {self.split_csv}")
            
        self.data = pd.read_csv(self.split_csv)
        
        # Enforce dummy labels if 'label' column is missing (for unsupervised or testing)
        if "label" not in self.data.columns:
            self.data["label"] = 0

    def __len__(self) -> int:
        return len(self.data)

    def _get_mask_path(self, img_path: Path) -> Path:
        stem = img_path.stem
        return self.mask_dir / f"{stem}.png"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.data.iloc[idx]
        img_rel_path = row["image_path"]
        label = int(row["label"])
        
        img_path = self.image_dir / img_rel_path
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        image = Image.open(img_path).convert("L")
        
        if self.mode in ["crop", "masked"]:
            if not self.mask_dir:
                raise ValueError("mask_dir must be provided for crop or masked mode.")
                
            mask_path = self._get_mask_path(Path(img_rel_path))
            if mask_path.exists():
                mask = Image.open(mask_path).convert("L")
                
                if self.mode == "masked":
                    from src.preprocessing.roi_processing import apply_mask
                    image = apply_mask(image, mask)
                elif self.mode == "crop":
                    from src.preprocessing.roi_processing import extract_roi
                    extracted = extract_roi(image, mask)
                    if extracted is not None:
                        image = extracted
                        
        if self.transform is not None:
            image = self.transform(image)
        else:
            from torchvision.transforms import ToTensor, Resize, Compose
            image = Compose([Resize((224, 224)), ToTensor()])(image)
            
        return image, label
