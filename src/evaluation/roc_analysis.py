"""ROC and Precision-Recall curve plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def plot_roc_curve(
    targets: list[int],
    probs: list[float],
    output_path: str | Path,
    label: str = "Model",
) -> None:
    """Plot and save a ROC curve."""
    fpr, tpr, _ = roc_curve(targets, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_precision_recall_curve(
    targets: list[int],
    probs: list[float],
    output_path: str | Path,
    label: str = "Model",
) -> None:
    """Plot and save a Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(targets, probs)
    pr_auc = average_precision_score(targets, probs)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, lw=2, label=f"{label} (PR-AUC = {pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
