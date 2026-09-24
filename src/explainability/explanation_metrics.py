"""Quantitative evaluation of XAI explanations against LV segmentation masks."""

from __future__ import annotations

import numpy as np


def xai_lv_energy_ratio(heatmap: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute what fraction of heatmap energy lies inside the LV mask.
    Energy_LV = sum(H[i,j] for (i,j) in LV) / sum(H)
    """
    heatmap = np.clip(heatmap.astype(np.float32), 0, None)
    binary_mask = (mask > 0).astype(np.float32)
    total = float(np.sum(heatmap))
    if total < 1e-8:
        return 0.0
    return float(np.sum(heatmap * binary_mask) / total)


def xai_lv_iou(heatmap: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> float:
    """Compute IoU between thresholded heatmap and LV mask."""
    binary_heat = (heatmap > threshold).astype(np.float32)
    binary_mask = (mask > 0).astype(np.float32)
    intersection = np.sum(binary_heat * binary_mask)
    union = np.sum(np.clip(binary_heat + binary_mask, 0, 1))
    return float(intersection / (union + 1e-6))


def categorize_xai_result(
    is_correct: bool,
    energy_ratio: float,
    energy_threshold: float = 0.5,
) -> str:
    """
    Classify a prediction+explanation result into one of four research categories:
      - correct_lv_focused
      - correct_irrelevant
      - incorrect_lv_focused
      - incorrect_irrelevant
    """
    focused = energy_ratio >= energy_threshold
    if is_correct and focused:
        return "correct_lv_focused"
    if is_correct and not focused:
        return "correct_irrelevant"
    if not is_correct and focused:
        return "incorrect_lv_focused"
    return "incorrect_irrelevant"


def summarize_xai_categories(categories: list[str]) -> dict[str, int]:
    """Count occurrences of each XAI result category."""
    keys = ["correct_lv_focused", "correct_irrelevant", "incorrect_lv_focused", "incorrect_irrelevant"]
    return {k: categories.count(k) for k in keys}
