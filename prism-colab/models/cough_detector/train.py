"""
PRISM Models — Cough Detector Training Loop

Full training loop with:
- BCEWithLogitsLoss (class-weight aware)
- Adam optimizer + CosineAnnealingLR scheduler
- Early stopping on validation loss
- Per-epoch metric tracking (AUC, F1, accuracy)
- Checkpoint saving for best model
- Rich progress bars and summary tables
"""

from __future__ import annotations

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

from models.shared.checkpoint import save_checkpoint
from models.shared.metrics import MetricTracker

console = Console()


class Trainer:
    """
    Trains the CoughDetector model.

    Args:
        model: CoughDetector instance
        train_loader: training DataLoader
        val_loader: validation DataLoader
        config: training config dict (from training.yaml)
        device: torch device
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Loss with class weighting
        pos_weight_val = getattr(train_loader.dataset, "pos_weight", 1.0)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_val], device=device)
        )

        # Optimizer
        lr = config.get("learning_rate", 0.001)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Scheduler
        epochs = config.get("epochs", 50)
        scheduler_cfg = config.get("scheduler", {})
        t_max = scheduler_cfg.get("T_max", epochs)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=t_max
        )

        # Early stopping
        self.patience = config.get("early_stopping_patience", 7)
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        # Checkpoint path
        checkpoint_dir = config.get("checkpoint_dir", "./models/checkpoints")
        self.checkpoint_path = Path(checkpoint_dir) / "cough_detector_best.pt"

        # Epochs
        self.total_epochs = epochs

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch, return metrics dict."""
        self.model.train()
        tracker = MetricTracker()

        for mel, labels in self.train_loader:
            mel = mel.to(self.device)
            labels = labels.float().to(self.device)

            # Forward pass — only use the cough logits, ignore embeddings
            logits, _ = self.model(mel)
            logits = logits.squeeze(-1)  # (B,)

            loss = self.criterion(logits, labels)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            tracker.update(loss.item(), logits, labels)

        return tracker.compute()

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        """Run validation, return metrics dict."""
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
        Full training loop.

        Returns:
            Best validation metrics dict.
        """
        console.print("\n[bold cyan]PRISM Cough Detector — Training[/bold cyan]\n")
        console.print(f"  Device        : {self.device}")
        console.print(f"  Epochs        : {self.total_epochs}")
        console.print(f"  Batch size    : {self.config.get('batch_size', 32)}")
        console.print(f"  Learning rate : {self.config.get('learning_rate', 0.001)}")
        console.print(f"  Train samples : {len(self.train_loader.dataset)}")  # type: ignore[arg-type]
        console.print(f"  Val samples   : {len(self.val_loader.dataset)}")  # type: ignore[arg-type]
        console.print(f"  Checkpoint    : {self.checkpoint_path}")
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
            task = progress.add_task("Training...", total=self.total_epochs)

            for epoch in range(1, self.total_epochs + 1):
                progress.update(task, description=f"Epoch {epoch}/{self.total_epochs}")

                # Train
                train_metrics = self._train_one_epoch(epoch)

                # Validate
                val_metrics = self._validate()

                # Step scheduler
                self.scheduler.step()

                # Log epoch summary
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

                    # Save best checkpoint
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

        # Print final summary table
        console.print()
        table = Table(
            title="Best Validation Metrics",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        for key, value in best_metrics.items():
            table.add_row(key.capitalize(), f"{value:.4f}")

        console.print(table)

        return best_metrics


def dry_run(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
) -> None:
    """
    Sanity check: run 1 forward + backward pass on a single batch.
    Verifies shapes, gradients, and loss computation.
    """
    console.print("\n[bold cyan]Dry Run — 1 Batch Sanity Check[/bold cyan]\n")

    model.train()
    mel, labels = next(iter(train_loader))
    mel = mel.to(device)
    labels = labels.float().to(device)

    console.print(f"  Input shape  : {mel.shape}")
    console.print(f"  Labels shape : {labels.shape}")
    console.print(f"  Labels       : {labels[:8].tolist()}")

    logits, embeddings = model(mel)
    console.print(f"  Logits shape : {logits.shape}")
    console.print(f"  Embed shape  : {embeddings.shape}")

    # Check embedding is L2-normalised
    norms = embeddings.norm(dim=1)
    console.print(f"  Embed norms  : min={norms.min():.4f}, max={norms.max():.4f}")

    # Loss
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits.squeeze(-1), labels)
    console.print(f"  Loss         : {loss.item():.4f}")

    # Backward
    loss.backward()
    grad_norm = sum(
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    )
    console.print(f"  Grad norm    : {grad_norm:.4f}")

    console.print(
        "\n[green]OK - Dry run passed: shapes, loss, and gradients OK[/green]"
    )
