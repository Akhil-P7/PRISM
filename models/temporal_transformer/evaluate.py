"""
PRISM Models — Temporal Transformer Evaluation

Standalone evaluation script that loads the best checkpoint and runs
inference on the test split.  Outputs:
    - Accuracy, Macro F1, Weighted F1, per-class precision/recall
    - Confusion matrix
    - Saves results to evaluation/temporal_analysis/temporal_eval.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from models.shared.checkpoint import load_checkpoint
from models.shared.metrics import MultiClassMetricTracker
from models.temporal_transformer.dataset import create_temporal_dataloaders
from models.temporal_transformer.model import TemporalTransformer, build_temporal_model

console = Console()

TRAJECTORY_LABELS = {
    0: "Stable",
    1: "Improving",
    2: "Increasing",
    3: "Abnormal",
}


@torch.no_grad()
def evaluate(
    model: TemporalTransformer,
    test_loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Run evaluation on the test set.

    Returns:
        Dict with metrics plus the confusion matrix and classification report.
    """
    model.eval()
    tracker = MultiClassMetricTracker(num_classes=4)

    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    criterion = torch.nn.CrossEntropyLoss()

    for sequences, labels in test_loader:
        sequences = sequences.to(device)
        labels = labels.long().to(device)

        logits = model(sequences)
        loss = criterion(logits, labels)
        tracker.update(loss.item(), logits, labels)

        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.append(preds.reshape(-1))
        all_labels.append(labels.cpu().numpy().reshape(-1))

    metrics = tracker.compute()

    # Confusion matrix
    all_preds_flat = np.concatenate(all_preds)
    all_labels_flat = np.concatenate(all_labels)
    cm = confusion_matrix(all_labels_flat, all_preds_flat)
    metrics["confusion_matrix"] = cm.tolist()

    # Per-class report
    target_names = [TRAJECTORY_LABELS[i] for i in range(4)]
    report = classification_report(
        all_labels_flat,
        all_preds_flat,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    metrics["per_class"] = report

    return metrics


def run_evaluation(
    checkpoint_path: str = "models/checkpoints/temporal_transformer_best.pt",
    data_dir: str = "datasets/temporal",
    output_path: str = "evaluation/temporal_analysis/temporal_eval.json",
) -> dict:
    """
    Full evaluation pipeline.

    Args:
        checkpoint_path: path to the best checkpoint .pt file
        data_dir: path to temporal dataset directory
        output_path: where to save the JSON results

    Returns:
        Dict with evaluation metrics.
    """
    console.print("\n[bold cyan]PRISM Temporal Transformer — Evaluation[/bold cyan]\n")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"  Device     : {device}")

    # Create test loader
    loaders = create_temporal_dataloaders(
        data_dir=data_dir,
        batch_size=64,
        num_workers=0,
    )
    test_loader = loaders["test"]
    console.print(f"  Test samples: {len(test_loader.dataset)}")  # type: ignore[arg-type]

    # Load model
    model = build_temporal_model(device=device)
    info = load_checkpoint(checkpoint_path, model, device=device)
    console.print(
        f"  Loaded checkpoint from epoch {info['epoch']} "
        f"(val Macro F1={info['metrics'].get('macro_f1', 'N/A')})"
    )

    # Evaluate
    console.print("\n  Running inference on test set...")
    metrics = evaluate(model, test_loader, device)

    # Print summary table
    console.print()
    table = Table(
        title="Test Set Results",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    for key in [
        "loss",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_precision",
        "macro_recall",
    ]:
        if key in metrics:
            table.add_row(key.replace("_", " ").title(), f"{metrics[key]:.4f}")

    console.print(table)

    # Per-class table
    console.print()
    class_table = Table(
        title="Per-Class Results",
        show_header=True,
        header_style="bold cyan",
    )
    class_table.add_column("Class", style="cyan")
    class_table.add_column("Precision", justify="right")
    class_table.add_column("Recall", justify="right")
    class_table.add_column("F1", justify="right")
    class_table.add_column("Support", justify="right")

    per_class = metrics.get("per_class", {})
    for class_name in TRAJECTORY_LABELS.values():
        if class_name in per_class:
            cls = per_class[class_name]
            class_table.add_row(
                class_name,
                f"{cls['precision']:.4f}",
                f"{cls['recall']:.4f}",
                f"{cls['f1-score']:.4f}",
                str(int(cls["support"])),
            )

    console.print(class_table)

    # Confusion matrix
    cm = metrics["confusion_matrix"]
    console.print("\n  Confusion Matrix:")
    header = "                " + "".join(
        f"{TRAJECTORY_LABELS[i]:>12}" for i in range(4)
    )
    console.print(header)
    for i in range(4):
        row_str = f"  {TRAJECTORY_LABELS[i]:<14}" + "".join(
            f"{cm[i][j]:>12}" for j in range(4)
        )
        console.print(row_str)

    # Save results
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    console.print(f"\n  Results saved to: {output}")

    return metrics


if __name__ == "__main__":
    run_evaluation()
