"""Single-image segmentation inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.utils.logger import get_logger
from src.segmentation.unet import UNet
from src.segmentation.unetplusplus import UNetPlusPlus

logger = get_logger("lv_xai.segmentation_inference")


def load_segmentation_model(
    checkpoint_path: str | Path,
    model_name: str = "unet",
    in_channels: int = 1,
    out_channels: int = 1,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Load a trained segmentation model from a weights file."""
    device = device or torch.device("cpu")
    if model_name == "unet++":
        model = UNetPlusPlus(in_channels, out_channels)
    else:
        model = UNet(in_channels, out_channels)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    logger.info("Loaded segmentation model from %s", checkpoint_path)
    return model


def predict_mask(
    model: torch.nn.Module,
    image: Image.Image,
    target_size: tuple[int, int] = (256, 256),
    threshold: float = 0.5,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    Run inference on a single PIL image and return a binary mask (H, W) as uint8.
    The mask is resized back to the original image dimensions.
    """
    device = device or torch.device("cpu")
    original_size = image.size  # (W, H)

    gray = image.convert("L")
    resized = gray.resize(target_size, Image.Resampling.BILINEAR)
    tensor = torch.tensor(np.array(resized), dtype=torch.float32) / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()

    binary = (prob > threshold).astype(np.uint8) * 255
    mask_pil = Image.fromarray(binary).resize(original_size, Image.Resampling.NEAREST)
    return np.array(mask_pil)
