"""Model checkpointing utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.utils.logger import get_logger

logger = get_logger("lv_xai.checkpoint")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    output_path: str | Path,
    config: dict[str, Any] | None = None,
) -> None:
    """Save a full training checkpoint including model weights, optimizer state, and metadata."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
    }
    torch.save(payload, output_path)
    logger.info("Checkpoint saved to %s (epoch %d)", output_path, epoch)


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Load model (and optionally optimizer) state from a checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    map_location = device or torch.device("cpu")
    payload = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    logger.info(
        "Loaded checkpoint from %s (epoch %d, metrics: %s)",
        checkpoint_path,
        payload.get("epoch", -1),
        payload.get("metrics", {}),
    )
    return payload


def save_metrics_json(metrics: dict[str, Any], output_path: str | Path) -> None:
    """Persist a metrics dictionary as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    logger.info("Metrics saved to %s", output_path)
