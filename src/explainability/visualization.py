"""XAI visualization utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def generate_overlay(
    image_arr: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Overlay a normalized heatmap onto a grayscale or RGB image.

    Args:
        image_arr: Input image as uint8 numpy array (H, W) or (H, W, 3).
        heatmap: Normalized heatmap in [0, 1] of shape (H, W).
        alpha: Blending weight for the heatmap.

    Returns:
        Blended RGB image as uint8 numpy array.
    """
    if image_arr.ndim == 2:
        image_arr = cv2.cvtColor(image_arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(
        image_arr.astype(np.uint8), 1 - alpha,
        heatmap_color.astype(np.uint8), alpha, 0
    )
    return overlay


def plot_gradcam(
    original_img: np.ndarray,
    mask: np.ndarray,
    heatmap: np.ndarray,
    target_class: int,
    output_path: str | Path,
    confidence: float | None = None,
) -> None:
    """
    Save a 4-panel Grad-CAM figure:
    [Original Image | LV Segmentation | Grad-CAM Heatmap | Overlay]
    """
    overlay = generate_overlay(original_img, heatmap)
    conf_str = f" ({confidence * 100:.1f}%)" if confidence is not None else ""
    titles = [
        "Original Image",
        "LV Segmentation",
        f"Grad-CAM (Class {target_class})",
        f"Overlay{conf_str}",
    ]
    panels = [original_img, mask, heatmap, overlay]
    cmaps = ["gray", "gray", "jet", None]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, panel, title, cmap in zip(axes, panels, titles, cmaps):
        ax.imshow(panel, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_shap_overlay(
    original_img: np.ndarray,
    attributions: np.ndarray,
    target_class: int,
    output_path: str | Path,
) -> None:
    """
    Save a 3-panel SHAP figure using a diverging colormap.
    Red pixels = positive attribution (push prediction toward class).
    Blue pixels = negative attribution (push prediction away from class).
    [Original | SHAP Map | Overlay]
    """
    vmax = float(np.abs(attributions).max()) + 1e-8

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_img, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    im = axes[1].imshow(attributions, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title(f"SHAP Attributions (Class {target_class})")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    if original_img.ndim == 2:
        base = cv2.cvtColor(original_img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        base = original_img.astype(np.uint8)
    norm_attr = (np.clip(attributions / vmax, -1, 1) + 1) / 2  # map to [0, 1]
    heat_uint8 = np.uint8(norm_attr * 255)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_COOL)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(base, 0.6, heat_color, 0.4, 0)

    axes[2].imshow(overlay)
    axes[2].set_title("SHAP Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
