"""Classification metric helpers."""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)


def compute_classification_metrics(
    targets: list[int],
    preds: list[int],
    probs: list[float] | None = None,
    num_classes: int = 2,
    average: str = "weighted",
) -> dict[str, float]:
    """
    Compute a comprehensive set of classification metrics.
    For binary classification (num_classes=2), also computes AUC and PR-AUC.
    """
    metrics: dict[str, float] = {
        "accuracy": accuracy_score(targets, preds),
        "precision": precision_score(targets, preds, average=average, zero_division=0),
        "recall": recall_score(targets, preds, average=average, zero_division=0),
        "f1": f1_score(targets, preds, average=average, zero_division=0),
    }
    if num_classes == 2 and probs is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(targets, probs)
            metrics["pr_auc"] = average_precision_score(targets, probs)
        except ValueError:
            metrics["roc_auc"] = 0.0
            metrics["pr_auc"] = 0.0
    return metrics


def get_classification_report(targets: list[int], preds: list[int]) -> str:
    """Return a formatted sklearn classification report string."""
    return classification_report(targets, preds, zero_division=0)
