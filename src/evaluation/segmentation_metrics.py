"""Segmentation metric helpers (standalone, dataset-level)."""

from __future__ import annotations

import numpy as np


def dice_from_arrays(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute Dice coefficient between two binary arrays."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    return float((2 * intersection + smooth) / (pred.sum() + target.sum() + smooth))


def iou_from_arrays(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute IoU (Jaccard index) between two binary arrays."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float((intersection + smooth) / (union + smooth))


def hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute the 95th-percentile Hausdorff distance between two binary masks.
    Uses scipy distance_transform_edt for efficiency.
    """
    from scipy.ndimage import distance_transform_edt

    pred = pred.astype(bool)
    target = target.astype(bool)

    if not pred.any() or not target.any():
        return float("inf")

    pred_dist = distance_transform_edt(~pred)
    target_dist = distance_transform_edt(~target)

    hd_pred_to_target = pred_dist[target].max()
    hd_target_to_pred = target_dist[pred].max()
    return float(max(hd_pred_to_target, hd_target_to_pred))


def compute_all_segmentation_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute Dice, IoU, and Hausdorff distance for a prediction/target pair."""
    return {
        "dice": dice_from_arrays(pred, target),
        "iou": iou_from_arrays(pred, target),
        "hausdorff": hausdorff_distance(pred, target),
    }
