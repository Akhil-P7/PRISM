"""
PRISM Models — Cough Detector Dataset

PyTorch Dataset that reads manifest.csv and loads pre-computed .npy
mel spectrograms.  Splits data by subject_id to prevent data leakage.

Usage::

    from models.cough_detector.dataset import CoughDataset, create_dataloaders

    loaders = create_dataloaders(
        manifest_path="datasets/features/manifest.csv",
        features_dir="datasets/features",
        batch_size=32,
    )
    for mel, label in loaders["train"]:
        ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from models.shared.transforms import Normalize, SpecAugment


class CoughDataset(Dataset):
    """
    PyTorch Dataset for binary cough detection.

    Each item is a tuple of:
        mel: FloatTensor shape (1, 128, T) — single-channel mel spectrogram
        label: int — 1 if cough, 0 otherwise
    """

    def __init__(
        self,
        df: pd.DataFrame,
        features_dir: str | Path,
        split: Literal["train", "val", "test"] = "train",
        augment: bool | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.features_dir = Path(features_dir)
        self.split = split

        # Augmentation: default on for train, off for val/test
        apply_augment = augment if augment is not None else (split == "train")
        self.normalize = Normalize()
        self.augment = SpecAugment() if apply_augment else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]

        # Load mel spectrogram
        mel_path = self.features_dir / row["mel_path"]
        mel = np.load(str(mel_path))  # (128, T)
        mel = torch.from_numpy(mel).float().unsqueeze(0)  # (1, 128, T)

        # Normalize
        mel = self.normalize(mel)

        # Augment (train only)
        if self.augment is not None:
            mel = self.augment(mel)

        # Label
        label = int(row["is_cough"])

        return mel, label

    @property
    def pos_weight(self) -> float:
        """Compute positive class weight for BCEWithLogitsLoss."""
        n_pos = int(self.df["is_cough"].sum())
        n_neg = len(self.df) - n_pos
        if n_pos == 0:
            return 1.0
        return n_neg / n_pos


def split_by_subject(
    manifest_path: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Split manifest.csv into train/val/test by subject_id.

    This prevents data leakage: no subject appears in more than one split.
    Silent segments are filtered out.

    Returns:
        Dict with keys "train", "val", "test", each containing a DataFrame.
    """
    df = pd.read_csv(manifest_path)

    # Filter out silent segments
    df = df[~df["is_silent"]].copy()

    # Get unique subjects
    subjects = df["subject_id"].unique()

    # Split subjects into train / (val+test)
    train_subjects, valtest_subjects = train_test_split(
        subjects, train_size=train_ratio, random_state=seed
    )

    # Split val+test into val / test (equal halves of remaining)
    relative_val = val_ratio / (1.0 - train_ratio)
    val_subjects, test_subjects = train_test_split(
        valtest_subjects, train_size=relative_val, random_state=seed
    )

    splits = {
        "train": df[df["subject_id"].isin(train_subjects)].copy(),
        "val": df[df["subject_id"].isin(val_subjects)].copy(),
        "test": df[df["subject_id"].isin(test_subjects)].copy(),
    }

    return splits


def create_dataloaders(
    manifest_path: str | Path = "datasets/features/manifest.csv",
    features_dir: str | Path = "datasets/features",
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool = True,
    weighted_sampling: bool = True,
) -> dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders from manifest.csv.

    Args:
        manifest_path: path to the manifest CSV
        features_dir: root dir containing mel/ and mfcc/ folders
        batch_size: batch size for all loaders
        num_workers: DataLoader worker count
        seed: random seed for split reproducibility
        pin_memory: pin memory for GPU transfer
        weighted_sampling: use WeightedRandomSampler for train loader

    Returns:
        Dict with keys "train", "val", "test" → DataLoader.
    """
    splits = split_by_subject(manifest_path, seed=seed)

    loaders: dict[str, DataLoader] = {}

    for split_name, df in splits.items():
        dataset = CoughDataset(
            df=df,
            features_dir=features_dir,
            split=split_name,  # type: ignore[arg-type]
        )

        sampler = None
        shuffle = split_name == "train"

        if split_name == "train" and weighted_sampling:
            # Weighted sampling to address class imbalance
            labels = df["is_cough"].astype(int).values
            class_counts = np.bincount(labels)
            weights = 1.0 / class_counts[labels]

            sampler = WeightedRandomSampler(
                weights=weights.tolist(),
                num_samples=len(labels),
                replacement=True,
            )
            shuffle = False  # mutually exclusive with sampler

        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split_name == "train"),
        )

    return loaders
