"""
PRISM Models — Temporal Transformer CLI Entry Point

Usage::

    # Full training run
    poetry run python -m models.temporal_transformer.run_training

    # Override config
    poetry run python -m models.temporal_transformer.run_training --epochs 50 --batch-size 32

    # 1-batch sanity check (no full training)
    poetry run python -m models.temporal_transformer.run_training --dry-run

    # Generate synthetic data first, then train
    poetry run python -m models.temporal_transformer.run_training --generate-data
"""

from __future__ import annotations

import argparse

import torch
import yaml  # type: ignore
from loguru import logger
from rich.console import Console

from models.temporal_transformer.dataset import create_temporal_dataloaders
from models.temporal_transformer.model import build_temporal_model
from models.temporal_transformer.train import TemporalTrainer, dry_run

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
    parser = argparse.ArgumentParser(
        description="PRISM Temporal Transformer — Training CLI"
    )
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
        "--data-dir",
        type=str,
        default="datasets/temporal",
        help="Path to temporal dataset directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 1-batch sanity check only",
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate synthetic temporal data before training",
    )

    args = parser.parse_args()

    # Load config
    config = load_config()
    temporal_cfg = config.get("temporal_transformer", {})
    training_cfg = config.get("training", {})

    # Apply CLI overrides
    if args.epochs is not None:
        temporal_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        temporal_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        temporal_cfg["learning_rate"] = args.lr

    # Merge checkpoint dir from general training config
    temporal_cfg["checkpoint_dir"] = training_cfg.get(
        "checkpoint_dir", "./models/checkpoints"
    )

    # Generate synthetic data if requested
    if args.generate_data:
        console.print("[bold]Generating synthetic temporal data...[/bold]")
        from models.temporal_transformer.generate_temporal_data import (
            generate_dataset,
            print_summary,
            split_and_save,
        )

        df = generate_dataset(patients_per_class=500, seed=training_cfg.get("seed", 42))
        print_summary(df)
        split_and_save(df, output_dir=args.data_dir, seed=training_cfg.get("seed", 42))

    # Resolve device
    device = get_device(training_cfg.get("device", "auto"))

    # Set seed
    seed = training_cfg.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    batch_size = temporal_cfg.get("batch_size", 16)
    num_workers = training_cfg.get("num_workers", 0)

    # On Windows, num_workers > 0 can cause multiprocessing deadlocks.
    if num_workers > 0:
        logger.warning(
            f"num_workers={num_workers} — setting to 0 for Windows compatibility"
        )
        num_workers = 0

    # Create data loaders
    console.print("[bold]Loading temporal dataset...[/bold]")
    loaders = create_temporal_dataloaders(
        data_dir=args.data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=training_cfg.get("pin_memory", True),
    )

    # Build model
    console.print("[bold]Building Temporal Transformer...[/bold]")
    model = build_temporal_model(config=temporal_cfg, device=device)

    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"  Parameters: {param_count:,} total, {trainable:,} trainable")

    # Dry run or full training
    if args.dry_run:
        dry_run(model, loaders["train"], device)
    else:
        trainer = TemporalTrainer(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            config=temporal_cfg,
            device=device,
        )
        trainer.train()


if __name__ == "__main__":
    main()
