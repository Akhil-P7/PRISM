"""
PRISM Models — Cough Detector Evaluation

Standalone evaluation script that loads the best checkpoint and runs
inference on the test split.  Outputs:
    - AUC-ROC, F1, precision, recall, accuracy
    - Confusion matrix
    - Saves results to evaluation/results/cough_detector_eval.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from models.cough_detector.dataset import create_dataloaders
from models.cough_detector.model import CoughDetector
from models.shared.checkpoint import load_checkpoint
from models.shared.metrics import MetricTracker

console = Console()


@torch.no_grad()
def evaluate(
    model: CoughDetector,
    test_loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Run evaluation on the test set.

    Returns:
        Dict with metrics plus the confusion matrix.
    """
    model.eval()
    tracker = MetricTracker()

    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    criterion = torch.nn.BCEWithLogitsLoss()

    for mel, labels in test_loader:
        mel = mel.to(device)
        labels_float = labels.float().to(device)

        logits, _ = model(mel)
        logits = logits.squeeze(-1)

        loss = criterion(logits, labels_float)
        tracker.update(loss.item(), logits, labels_float)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs.reshape(-1))
        all_labels.append(labels.numpy().reshape(-1))

    metrics = tracker.compute()

    # Confusion matrix
    all_probs_flat = np.concatenate(all_probs)
    all_labels_flat = np.concatenate(all_labels)
    preds = (all_probs_flat >= 0.5).astype(int)
    cm = confusion_matrix(all_labels_flat, preds)

    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def run_evaluation(
    checkpoint_path: str = "models/checkpoints/cough_detector_best.pt",
    manifest_path: str = "datasets/features/manifest.csv",
    features_dir: str = "datasets/features",
    output_path: str = "evaluation/results/cough_detector_eval.json",
) -> dict:
    """
    Full evaluation pipeline.

    Args:
        checkpoint_path: path to the best checkpoint .pt file
        manifest_path: path to manifest.csv
        features_dir: path to features directory
        output_path: where to save the JSON results

    Returns:
        Dict with evaluation metrics.
    """
    console.print("\n[bold cyan]PRISM Cough Detector — Evaluation[/bold cyan]\n")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"  Device     : {device}")

    # Create test loader
    loaders = create_dataloaders(
        manifest_path=manifest_path,
        features_dir=features_dir,
        batch_size=64,
        num_workers=0,
        weighted_sampling=False,
    )
    test_loader = loaders["test"]
    console.print(f"  Test samples: {len(test_loader.dataset)}")  # type: ignore[arg-type]

    # Load model
    model = CoughDetector()
    model = model.to(device)
    info = load_checkpoint(checkpoint_path, model, device=device)
    console.print(
        f"  Loaded checkpoint from epoch {info['epoch']} "
        f"(val AUC={info['metrics'].get('auc', 'N/A')})"
    )

    # Evaluate
    console.print("\n  Running inference on test set...")
    metrics = evaluate(model, test_loader, device)

    # Print results table
    console.print()
    table = Table(
        title="Test Set Results",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    for key in ["loss", "accuracy", "auc", "f1", "precision", "recall"]:
        table.add_row(key.upper(), f"{metrics[key]:.4f}")

    console.print(table)

    # Confusion matrix
    cm = metrics["confusion_matrix"]
    console.print("\n  Confusion Matrix:")
    console.print("                 Predicted 0   Predicted 1")
    console.print(f"  Actual 0       {cm[0][0]:>10}   {cm[0][1]:>10}")
    console.print(f"  Actual 1       {cm[1][0]:>10}   {cm[1][1]:>10}")

    # Save results
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    console.print(f"\n  Results saved to: {output}")

    return metrics


if __name__ == "__main__":
    run_evaluation()
