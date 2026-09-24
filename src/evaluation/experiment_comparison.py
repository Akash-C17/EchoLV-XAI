"""XAI vs LV-mask quantitative evaluation and experiment comparison."""

from __future__ import annotations

import numpy as np


def calculate_xai_lv_overlap(
    heatmap: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Quantify how much of the XAI heatmap energy falls within the LV segmentation mask.

    Implements the Energy_LV metric:
        Energy_LV = sum(H[i,j] for (i,j) in LV) / sum(H)

    Args:
        heatmap: Normalized attribution map in [0, 1], shape (H, W).
        mask: Binary LV segmentation mask (any non-zero value = LV), shape (H, W).
        threshold: Threshold to binarize heatmap for IoU computation.

    Returns:
        Dictionary containing iou, energy_in_lv, energy_outside_lv, energy_ratio.
    """
    heatmap = heatmap.astype(np.float32)
    binary_heat = (heatmap > threshold).astype(np.float32)
    binary_mask = (mask > 0).astype(np.float32)

    intersection = np.sum(binary_heat * binary_mask)
    union = np.sum(np.clip(binary_heat + binary_mask, 0, 1))
    iou = float(intersection / (union + 1e-6))

    total_energy = float(np.sum(heatmap))
    energy_in_lv = float(np.sum(heatmap * binary_mask))
    energy_outside_lv = total_energy - energy_in_lv
    energy_ratio = energy_in_lv / (total_energy + 1e-6)

    return {
        "iou": iou,
        "energy_in_lv": energy_in_lv,
        "energy_outside_lv": energy_outside_lv,
        "energy_ratio": energy_ratio,
    }


def analyze_errors(
    preds: list[int],
    targets: list[int],
    confidences: list[float],
    image_paths: list[str],
) -> list[dict]:
    """
    Categorize prediction errors into FP/FN with confidence scores.
    Returns a list sorted by descending confidence (highest-confidence errors first).
    """
    errors = []
    for p, t, conf, path in zip(preds, targets, confidences, image_paths):
        if p != t:
            errors.append({
                "path": str(path),
                "predicted": p,
                "target": t,
                "confidence": conf,
                "error_type": "FP" if p == 1 and t == 0 else "FN",
            })
    return sorted(errors, key=lambda x: x["confidence"], reverse=True)


def build_comparison_table(
    results: dict[str, dict[str, float]],
) -> str:
    """
    Format a comparison table of results across experimental modes.

    Args:
        results: {"full": {"accuracy": 0.9, "f1": 0.88, ...}, "crop": {...}, ...}

    Returns:
        A formatted ASCII table string.
    """
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    header = f"{'Model':<20}" + "".join(f"{m:>12}" for m in metrics)
    separator = "-" * len(header)
    rows = [header, separator]
    for mode, vals in results.items():
        row = f"{mode:<20}" + "".join(
            f"{vals.get(m, 0.0):>12.4f}" for m in metrics
        )
        rows.append(row)
    return "\n".join(rows)
