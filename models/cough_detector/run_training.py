"""
PRISM Models — Cough Detector CLI Entry Point

Usage::

    # Full training run
    poetry run python -m models.cough_detector.run_training

    # Override config
    poetry run python -m models.cough_detector.run_training --epochs 10 --batch-size 64

    # 1-batch sanity check (no full training)
    poetry run python -m models.cough_detector.run_training --dry-run
"""

from __future__ import annotations

import argparse

import torch
import yaml  # type: ignore
from loguru import logger
from rich.console import Console

from models.cough_detector.dataset import create_dataloaders
from models.cough_detector.model import build_model
from models.cough_detector.train import Trainer, dry_run

console = Console()


def get_device(preference: str = "auto") -> torch.device:
    """Resolve the training device from a config string."""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def load_config(path: str = "configs/training.yaml") -> dict:
    """Load training configuration from YAML."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="PRISM Cough Detector — Training CLI")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="datasets/features/manifest.csv",
        help="Path to manifest.csv",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default="datasets/features",
        help="Path to features directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 1-batch sanity check only",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Don't use ImageNet pretrained weights",
    )

    args = parser.parse_args()

    # Load config
    config = load_config()
    cough_cfg = config.get("cough_detector", {})
    training_cfg = config.get("training", {})

    # Apply CLI overrides
    if args.epochs is not None:
        cough_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cough_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        cough_cfg["learning_rate"] = args.lr

    # Merge checkpoint dir from general training config
    cough_cfg["checkpoint_dir"] = training_cfg.get(
        "checkpoint_dir", "./models/checkpoints"
    )

    # Resolve device
    device = get_device(training_cfg.get("device", "auto"))

    # Set seed
    seed = training_cfg.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    batch_size = cough_cfg.get("batch_size", 32)
    num_workers = training_cfg.get("num_workers", 0)

    # On Windows, num_workers > 0 can cause multiprocessing deadlocks.
    # We must force it to 0 for safety.
    if num_workers > 0:
        logger.warning(
            f"num_workers={num_workers} — setting to 0 for Windows compatibility"
        )
        num_workers = 0

    # Create data loaders
    console.print("[bold]Loading dataset...[/bold]")
    loaders = create_dataloaders(
        manifest_path=args.manifest,
        features_dir=args.features_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        pin_memory=training_cfg.get("pin_memory", True),
    )

    # Build model
    console.print("[bold]Building model...[/bold]")
    embedding_dim = config.get("embeddings", {}).get("dimension", 512)
    model = build_model(
        num_classes=cough_cfg.get("num_classes", 1),
        embedding_dim=embedding_dim,
        pretrained=not args.no_pretrained,
        device=device,
    )

    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"  Parameters: {param_count:,} total, {trainable:,} trainable")

    # Dry run or full training
    if args.dry_run:
        dry_run(model, loaders["train"], device)
    else:
        trainer = Trainer(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            config=cough_cfg,
            device=device,
        )
        trainer.train()


if __name__ == "__main__":
    main()
