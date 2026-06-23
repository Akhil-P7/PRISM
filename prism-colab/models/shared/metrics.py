"""
PRISM Models — Shared Metrics Tracker

Running accumulator for training/validation metrics across batches.
Tracks: loss, accuracy, AUC-ROC, F1, precision, recall.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class MetricTracker:
    """
    Accumulates predictions and labels across batches, then computes
    aggregate metrics at the end of an epoch.

    Usage::

        tracker = MetricTracker()
        for batch in loader:
            loss, logits, labels = ...
            tracker.update(loss.item(), logits, labels)
        metrics = tracker.compute()
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._losses: list[float] = []
        self._all_probs: list[np.ndarray] = []
        self._all_preds: list[np.ndarray] = []
        self._all_labels: list[np.ndarray] = []
        self._n_samples: int = 0

    def update(
        self,
        loss: float,
        logits: torch.Tensor,  # noqa: F821
        labels: torch.Tensor,  # noqa: F821
    ) -> None:
        """
        Args:
            loss: scalar loss value for this batch
            logits: raw model output tensor, shape (B,) or (B, 1)
            labels: ground-truth binary labels, shape (B,)
        """
        import torch

        logits = logits.detach().cpu().squeeze()
        labels = labels.detach().cpu()

        probs = torch.sigmoid(logits).numpy()
        preds = (probs >= 0.5).astype(int)
        labs = labels.numpy().astype(int)

        batch_size = len(labs)
        self._losses.append(loss * batch_size)
        self._n_samples += batch_size
        self._all_probs.append(probs.reshape(-1))
        self._all_preds.append(preds.reshape(-1))
        self._all_labels.append(labs.reshape(-1))

    def compute(self) -> dict[str, float]:
        """Return dict of aggregated metrics for the epoch."""
        if self._n_samples == 0:
            return {"loss": 0.0, "accuracy": 0.0, "auc": 0.0, "f1": 0.0}

        all_probs = np.concatenate(self._all_probs)
        all_preds = np.concatenate(self._all_preds)
        all_labels = np.concatenate(self._all_labels)

        avg_loss = sum(self._losses) / self._n_samples

        # Guard against single-class batches (edge case in tiny splits)
        try:
            auc = float(roc_auc_score(all_labels, all_probs))
        except ValueError:
            auc = 0.0

        return {
            "loss": round(avg_loss, 5),
            "accuracy": round(float(accuracy_score(all_labels, all_preds)), 4),
            "auc": round(auc, 4),
            "f1": round(float(f1_score(all_labels, all_preds, zero_division=0)), 4),
            "precision": round(
                float(precision_score(all_labels, all_preds, zero_division=0)), 4
            ),
            "recall": round(
                float(recall_score(all_labels, all_preds, zero_division=0)), 4
            ),
        }


class MultiClassMetricTracker:
    """
    Accumulates predictions and labels for multi-class classification,
    then computes aggregate metrics at epoch end.

    Used by the Temporal Transformer trainer (4-class trajectory prediction).

    Usage::

        tracker = MultiClassMetricTracker(num_classes=4)
        for batch in loader:
            loss, logits, labels = ...
            tracker.update(loss.item(), logits, labels)
        metrics = tracker.compute()
    """

    def __init__(self, num_classes: int = 4) -> None:
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self._losses: list[float] = []
        self._all_preds: list[np.ndarray] = []
        self._all_labels: list[np.ndarray] = []
        self._n_samples: int = 0

    def update(
        self,
        loss: float,
        logits: torch.Tensor,  # noqa: F821
        labels: torch.Tensor,  # noqa: F821
    ) -> None:
        """
        Args:
            loss: scalar loss value for this batch
            logits: raw model output, shape (B, num_classes)
            labels: ground-truth class indices, shape (B,)
        """
        import torch  # noqa: F811

        logits = logits.detach().cpu()
        labels = labels.detach().cpu()

        preds = torch.argmax(logits, dim=-1).numpy()
        labs = labels.numpy().astype(int)

        batch_size = len(labs)
        self._losses.append(loss * batch_size)
        self._n_samples += batch_size
        self._all_preds.append(preds.reshape(-1))
        self._all_labels.append(labs.reshape(-1))

    def compute(self) -> dict[str, float]:
        """Return dict of aggregated multi-class metrics for the epoch."""
        if self._n_samples == 0:
            return {
                "loss": 0.0,
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "weighted_f1": 0.0,
            }

        all_preds = np.concatenate(self._all_preds)
        all_labels = np.concatenate(self._all_labels)

        avg_loss = sum(self._losses) / self._n_samples

        return {
            "loss": round(avg_loss, 5),
            "accuracy": round(float(accuracy_score(all_labels, all_preds)), 4),
            "macro_f1": round(
                float(
                    f1_score(all_labels, all_preds, average="macro", zero_division=0)
                ),
                4,
            ),
            "weighted_f1": round(
                float(
                    f1_score(all_labels, all_preds, average="weighted", zero_division=0)
                ),
                4,
            ),
            "macro_precision": round(
                float(
                    precision_score(
                        all_labels, all_preds, average="macro", zero_division=0
                    )
                ),
                4,
            ),
            "macro_recall": round(
                float(
                    recall_score(
                        all_labels, all_preds, average="macro", zero_division=0
                    )
                ),
                4,
            ),
        }
