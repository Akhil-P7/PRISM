"""
PRISM Models — CNN Cough Detector Fine-Tuning

Fine-tunes the pre-trained CoughDetector (ResNet-18) using
waveform-domain augmentation to make it robust to browser-mic audio.

Key differences from original training:
    - Loads existing checkpoint as starting point
    - Lower learning rate (1e-4 vs 1e-3)
    - Freezes early layers for first N epochs
    - Uses FineTuneDataset with MicAugment
    - Saves to a SEPARATE checkpoint (never overwrites original)

Usage::

    # Full fine-tuning
    poetry run python -m models.cough_detector.finetune

    # Quick sanity check
    poetry run python -m models.cough_detector.finetune --dry-run

    # Override parameters
    poetry run python -m models.cough_detector.finetune --epochs 20 --lr 5e-5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch.utils.data import DataLoader

from models.cough_detector.finetune_dataset import create_finetune_loaders
from models.cough_detector.model import CoughDetector, build_model
from models.shared.checkpoint import load_checkpoint, save_checkpoint
from models.shared.metrics import MetricTracker

console = Console()


# ──────────────────────────────────────────────────────────────────
# Layer freezing
# ──────────────────────────────────────────────────────────────────


def freeze_early_layers(model: CoughDetector) -> int:
    """
    Freeze early ResNet layers (conv1, bn1, layer1, layer2).

    These layers learn low-level audio features (edges, textures)
    that are domain-agnostic and don't need to change.

    Returns:
        Number of frozen parameters.
    """
    frozen_modules = [model.conv1, model.bn1, model.layer1, model.layer2]
    frozen_count = 0

    for module in frozen_modules:
        for param in module.parameters():
            param.requires_grad = False
            frozen_count += param.numel()

    return frozen_count


def unfreeze_all(model: CoughDetector) -> None:
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


# ──────────────────────────────────────────────────────────────────
# Fine-Tune Trainer
# ──────────────────────────────────────────────────────────────────


class FineTuneTrainer:
    """
    Fine-tuning trainer with layer freezing schedule.

    Freezes early layers for the first `freeze_epochs`, then
    unfreezes everything for remaining epochs.
    """

    def __init__(
        self,
        model: CoughDetector,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 1e-4,
        epochs: int = 15,
        freeze_epochs: int = 5,
        patience: int = 5,
        checkpoint_path: str | Path = "models/checkpoints/cough_detector_finetuned.pt",
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.total_epochs = epochs
        self.freeze_epochs = freeze_epochs
        self.patience = patience
        self.checkpoint_path = Path(checkpoint_path)

        # Loss with class weighting
        pos_weight_val = getattr(train_loader.dataset, "pos_weight", 1.0)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_val], device=device)
        )

        # Optimizer — only non-frozen parameters
        self.lr = lr
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

        # Early stopping
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

    def _train_one_epoch(self) -> dict[str, float]:
        """Train one epoch."""
        self.model.train()
        tracker = MetricTracker()

        for mel, labels in self.train_loader:
            mel = mel.to(self.device)
            labels = labels.float().to(self.device)

            logits, _ = self.model(mel)
            logits = logits.squeeze(-1)
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            tracker.update(loss.item(), logits, labels)

        return tracker.compute()

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        """Validate on the val set."""
        self.model.eval()
        tracker = MetricTracker()

        for mel, labels in self.val_loader:
            mel = mel.to(self.device)
            labels = labels.float().to(self.device)

            logits, _ = self.model(mel)
            logits = logits.squeeze(-1)
            loss = self.criterion(logits, labels)

            tracker.update(loss.item(), logits, labels)

        return tracker.compute()

    def train(self) -> dict[str, float]:
        """
        Full fine-tuning loop with layer freezing schedule.

        Returns:
            Best validation metrics.
        """
        console.print("\n[bold cyan]PRISM Cough Detector — Fine-Tuning[/bold cyan]\n")
        console.print(f"  Device         : {self.device}")
        console.print(f"  Epochs         : {self.total_epochs}")
        console.print(f"  Freeze epochs  : {self.freeze_epochs}")
        console.print(f"  Learning rate  : {self.lr}")
        console.print(f"  Patience       : {self.patience}")
        console.print(f"  Train samples  : {len(self.train_loader.dataset)}")
        console.print(f"  Val samples    : {len(self.val_loader.dataset)}")
        console.print(f"  Checkpoint     : {self.checkpoint_path}")
        console.print()

        best_metrics: dict[str, float] = {}

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )

        with progress:
            task = progress.add_task("Fine-tuning...", total=self.total_epochs)

            for epoch in range(1, self.total_epochs + 1):
                progress.update(task, description=f"Epoch {epoch}/{self.total_epochs}")

                # Layer freezing schedule
                if epoch == 1:
                    frozen = freeze_early_layers(self.model)
                    trainable = sum(
                        p.numel() for p in self.model.parameters() if p.requires_grad
                    )
                    console.print(
                        f"  [dim]Frozen {frozen:,} params, "
                        f"training {trainable:,} params[/dim]"
                    )

                if epoch == self.freeze_epochs + 1:
                    unfreeze_all(self.model)
                    # Rebuild optimizer with all parameters
                    self.optimizer = torch.optim.Adam(
                        self.model.parameters(), lr=self.lr * 0.1
                    )
                    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer,
                        T_max=self.total_epochs - self.freeze_epochs,
                    )
                    trainable = sum(
                        p.numel() for p in self.model.parameters() if p.requires_grad
                    )
                    console.print(
                        f"\n  [bold green]Unfreezing all layers — "
                        f"now training {trainable:,} params at LR={self.lr * 0.1}[/bold green]\n"
                    )

                # Train
                train_metrics = self._train_one_epoch()

                # Validate
                val_metrics = self._validate()

                # Step scheduler
                self.scheduler.step()

                # Log
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss: {train_metrics['loss']:.4f}  "
                    f"AUC: {train_metrics['auc']:.3f}  "
                    f"F1: {train_metrics['f1']:.3f} | "
                    f"Val Loss: {val_metrics['loss']:.4f}  "
                    f"AUC: {val_metrics['auc']:.3f}  "
                    f"F1: {val_metrics['f1']:.3f} | "
                    f"LR: {lr:.6f}"
                )

                # Early stopping check
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.epochs_without_improvement = 0
                    best_metrics = val_metrics.copy()

                    save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        metrics=val_metrics,
                        path=self.checkpoint_path,
                    )
                else:
                    self.epochs_without_improvement += 1
                    if self.epochs_without_improvement >= self.patience:
                        console.print(
                            f"\n[yellow]Early stopping at epoch {epoch} "
                            f"(no improvement for {self.patience} epochs)[/yellow]"
                        )
                        break

                progress.advance(task)

        # Final summary
        console.print()
        table = Table(
            title="Best Fine-Tuned Validation Metrics",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        for key, value in best_metrics.items():
            table.add_row(key.capitalize(), f"{value:.4f}")

        console.print(table)

        return best_metrics


# ──────────────────────────────────────────────────────────────────
# Dry run
# ──────────────────────────────────────────────────────────────────


def dry_run(
    model: CoughDetector,
    train_loader: DataLoader,
    device: torch.device,
) -> None:
    """1-batch sanity check for the fine-tuning pipeline."""
    console.print("\n[bold cyan]Fine-Tune Dry Run — 1 Batch[/bold cyan]\n")

    model.train()
    frozen = freeze_early_layers(model)
    console.print(f"  Frozen params: {frozen:,}")

    mel, labels = next(iter(train_loader))
    mel = mel.to(device)
    labels = labels.float().to(device)

    console.print(f"  Input shape  : {mel.shape}")
    console.print(f"  Labels       : {labels[:8].tolist()}")

    logits, embeddings = model(mel)
    console.print(f"  Logits shape : {logits.shape}")
    console.print(f"  Embed shape  : {embeddings.shape}")

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits.squeeze(-1), labels)
    console.print(f"  Loss         : {loss.item():.4f}")

    loss.backward()
    grad_norm = sum(
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    )
    no_grad = sum(
        1 for p in model.parameters() if p.grad is None or not p.requires_grad
    )
    console.print(f"  Grad norm    : {grad_norm:.4f}")
    console.print(f"  Frozen layers: {no_grad} param groups with no gradient")

    console.print("\n[green]OK - Fine-tune dry run passed[/green]")


# ──────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PRISM Cough Detector — Fine-Tuning for Mic Robustness"
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--augment-prob", type=float, default=0.8)
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        default="models/checkpoints/cough_detector_best.pt",
        help="Pre-trained checkpoint to fine-tune from",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="datasets/features/manifest.csv",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default="datasets/features",
    )
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Seed
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Create data loaders with mic augmentation
    console.print("[bold]Loading fine-tune dataset...[/bold]")
    loaders = create_finetune_loaders(
        manifest_path=args.manifest,
        features_dir=args.features_dir,
        batch_size=args.batch_size,
        num_workers=0,  # Windows compatibility
        augment_prob=args.augment_prob,
    )

    # Build model and load pre-trained checkpoint
    console.print("[bold]Loading pre-trained model...[/bold]")
    model = build_model(pretrained=False, device=device)

    if Path(args.base_checkpoint).exists():
        info = load_checkpoint(args.base_checkpoint, model, device=device)
        console.print(
            f"  Loaded base checkpoint: epoch {info['epoch']}, "
            f"AUC={info['metrics'].get('auc', 'N/A')}"
        )
    else:
        console.print(
            f"  [yellow]Warning: Base checkpoint not found at {args.base_checkpoint}. "
            f"Fine-tuning from random weights.[/yellow]"
        )

    total_params = sum(p.numel() for p in model.parameters())
    console.print(f"  Total parameters: {total_params:,}")

    if args.dry_run:
        dry_run(model, loaders["train"], device)
    else:
        trainer = FineTuneTrainer(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            device=device,
            lr=args.lr,
            epochs=args.epochs,
            freeze_epochs=args.freeze_epochs,
            patience=args.patience,
        )
        best = trainer.train()

        # Save evaluation results
        eval_path = Path("models/checkpoints/finetune_eval.json")
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_path, "w") as f:
            json.dump(best, f, indent=2)
        console.print(f"\n  Eval metrics saved to: {eval_path}")


if __name__ == "__main__":
    main()
