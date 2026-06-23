"""
PRISM Models — Checkpoint Save/Load Utilities

Saves and loads model checkpoints with optimizer state, epoch,
and config metadata in a single .pt file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from loguru import logger


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    path: str | Path,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """
    Save a training checkpoint.

    Args:
        model: the model to save
        optimizer: optimizer with current state
        epoch: current epoch number
        metrics: dict of metrics (e.g. loss, auc) for reference
        path: file path for the checkpoint
        extra: optional additional metadata to include

    Returns:
        The path the checkpoint was written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if extra:
        checkpoint["extra"] = extra

    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved: {path} (epoch {epoch})")
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Load a training checkpoint.

    Args:
        path: path to the checkpoint file
        model: model to load state into
        optimizer: optional optimizer to restore state
        device: device to map tensors to

    Returns:
        Dict with 'epoch', 'metrics', and optional 'extra'.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_location = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Model loaded from: {path} (epoch {checkpoint['epoch']})")

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "epoch": checkpoint["epoch"],
        "metrics": checkpoint.get("metrics", {}),
        "extra": checkpoint.get("extra", {}),
    }
