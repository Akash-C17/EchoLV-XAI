"""General-purpose visualization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def save_sample_grid(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    output_path: str | Path,
    cmap: str = "gray",
    figsize_per_col: tuple[int, int] = (4, 4),
) -> None:
    """Save a row of images as a single figure."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_col[0] * n, figsize_per_col[1]))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap=cmap if img.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(
    history: list[dict],
    metric_pairs: list[tuple[str, str]],
    output_path: str | Path,
) -> None:
    """
    Plot training/validation curves for a list of metric pairs.
    metric_pairs: e.g. [("train_loss", "val_loss"), ("train_dice", "val_dice")]
    """
    n = len(metric_pairs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]
    epochs = [row["epoch"] for row in history]
    for ax, (train_key, val_key) in zip(axes, metric_pairs):
        train_vals = [row[train_key] for row in history]
        val_vals = [row[val_key] for row in history]
        ax.plot(epochs, train_vals, label="Train")
        ax.plot(epochs, val_vals, label="Val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(train_key.replace("train_", "").replace("_", " ").title())
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
