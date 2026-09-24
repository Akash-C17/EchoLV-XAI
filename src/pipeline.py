"""End-to-end inference pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.segmentation.inference import load_segmentation_model, predict_mask
from src.classification.inference import load_classification_model, predict
from src.preprocessing.roi_processing import extract_roi, apply_mask
from src.explainability.gradcam import GradCAM, get_target_layer
from src.explainability.visualization import generate_overlay

logger = get_logger("lv_xai.pipeline")


class InferencePipeline:
    """
    Full end-to-end inference pipeline:
    Image -> Preprocessing -> U-Net Segmentation -> ROI Extraction
           -> CNN Classification -> Grad-CAM -> Result
    """

    def __init__(
        self,
        config_path: str | Path = "config.yaml",
        seg_checkpoint: str | Path | None = None,
        cls_checkpoint: str | Path | None = None,
        cls_mode: str = "crop",
        class_names: list[str] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = load_config(config_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cls_mode = cls_mode
        self.class_names = class_names

        seg_cfg = self.config.values["segmentation"]
        cls_cfg = self.config.values["classification"]

        # Load segmentation model
        seg_ckpt = seg_checkpoint or (
            self.config.path("models") / "segmentation" / f"best_{seg_cfg['model']}.pth"
        )
        self.seg_model = load_segmentation_model(
            seg_ckpt,
            model_name=seg_cfg["model"],
            in_channels=seg_cfg["input_channels"],
            out_channels=seg_cfg["output_channels"],
            device=self.device,
        )

        # Load classification model
        cls_ckpt = cls_checkpoint or (
            self.config.path("models") / "classification" / f"best_{cls_mode}_resnet18.pth"
        )
        self.cls_model = load_classification_model(
            cls_ckpt,
            num_classes=cls_cfg["num_classes"],
            device=self.device,
        )

        # Grad-CAM setup
        target_layer = get_target_layer(self.cls_model, "resnet18")
        self.gradcam = GradCAM(self.cls_model, target_layer)

        logger.info(
            "Pipeline initialized (mode=%s, device=%s)", cls_mode, self.device
        )

    def predict(self, image_path: str | Path) -> dict:
        """
        Run the full pipeline on a single echocardiographic image.

        Args:
            image_path: Path to the input image file.

        Returns:
            Dictionary with keys:
                predicted_class, predicted_label, confidence,
                segmentation_mask, roi, gradcam_heatmap, gradcam_overlay.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        original = Image.open(image_path).convert("L")
        seg_size = self.config.values["segmentation"].get("segmentation_image_size", 256)
        cls_size = self.config.values["data"]["image_size"]
        threshold = self.config.values["segmentation"]["threshold"]

        # 1. Segmentation
        mask_arr = predict_mask(
            self.seg_model, original,
            target_size=(seg_size, seg_size),
            threshold=threshold,
            device=self.device,
        )
        mask_pil = Image.fromarray(mask_arr)

        # 2. ROI / Masking based on mode
        if self.cls_mode == "crop":
            input_img = extract_roi(original, mask_pil) or original
        elif self.cls_mode == "masked":
            input_img = apply_mask(original, mask_pil)
        else:
            input_img = original

        # 3. Classification
        cls_result = predict(
            self.cls_model, input_img,
            target_size=(cls_size, cls_size),
            device=self.device,
            class_names=self.class_names,
        )

        # 4. Grad-CAM on the classification input
        arr = np.array(input_img.resize((cls_size, cls_size))) / 255.0
        tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        heatmap, _ = self.gradcam(tensor, target_class=cls_result["predicted_class"])

        img_uint8 = np.array(input_img.resize((cls_size, cls_size))).astype(np.uint8)
        overlay = generate_overlay(img_uint8, heatmap)

        return {
            "predicted_class": cls_result["predicted_class"],
            "predicted_label": cls_result["predicted_label"],
            "confidence": cls_result["confidence"],
            "probabilities": cls_result["probabilities"],
            "segmentation_mask": mask_arr,
            "roi": np.array(input_img),
            "gradcam_heatmap": heatmap,
            "gradcam_overlay": overlay,
        }
