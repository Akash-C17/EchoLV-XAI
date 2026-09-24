"""Single-image classification inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.classification.models import EchoClassifier
from src.utils.logger import get_logger

logger = get_logger("lv_xai.classification_inference")


def load_classification_model(
    checkpoint_path: str | Path,
    num_classes: int = 2,
    in_channels: int = 1,
    device: torch.device | None = None,
) -> EchoClassifier:
    """Load a trained EchoClassifier from a weights file."""
    device = device or torch.device("cpu")
    model = EchoClassifier(num_classes=num_classes, pretrained=False, in_channels=in_channels)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    logger.info("Loaded classification model from %s", checkpoint_path)
    return model


def predict(
    model: EchoClassifier,
    image: Image.Image,
    target_size: tuple[int, int] = (224, 224),
    device: torch.device | None = None,
    class_names: list[str] | None = None,
) -> dict:
    """
    Run inference on a single PIL image.
    Returns: dict with predicted_class, predicted_label, confidence, probabilities.
    """
    device = device or torch.device("cpu")

    gray = image.convert("L")
    resized = gray.resize(target_size, Image.Resampling.BILINEAR)
    tensor = torch.tensor(np.array(resized), dtype=torch.float32) / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])
    label = class_names[pred_class] if class_names else str(pred_class)

    return {
        "predicted_class": pred_class,
        "predicted_label": label,
        "confidence": confidence,
        "probabilities": probs.tolist(),
    }
