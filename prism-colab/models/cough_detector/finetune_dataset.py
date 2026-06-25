"""
PRISM Models — Fine-Tune Dataset for Mic Robustness

PyTorch Dataset that loads pre-computed mel spectrograms, inverts them
back to waveforms, applies MicAugment waveform-domain transformations,
then recomputes the mel spectrogram.

This teaches the CNN what coughs sound like through noisy, reverberant,
codec-degraded browser-microphone recordings — without needing any new data.

Usage::

    from models.cough_detector.finetune_dataset import create_finetune_loaders

    loaders = create_finetune_loaders(
        manifest_path="datasets/features/manifest.csv",
        features_dir="datasets/features",
        batch_size=64,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from models.shared.transforms import Normalize, SpecAugment

# Audio config (must match training.yaml)
SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
F_MIN = 20
F_MAX = 8000


class FastMicAugment:
    """Mic-simulation augmentation directly on mel spectrograms (no waveform inversion)."""

    def __init__(self, p=0.8):
        self.p = p

    def __call__(self, mel: np.ndarray) -> np.ndarray:
        import random

        from scipy import signal as scipy_signal

        if random.random() > self.p:
            return mel
        result = mel.copy()

        # 1. Background noise (additive noise on spectrogram)
        if random.random() < 0.7:
            snr = random.uniform(5, 25)
            noise = np.random.randn(*result.shape).astype(np.float32)
            noise_scale = np.std(result) / (10 ** (snr / 20) + 1e-8)
            result = result + noise * noise_scale

        # 2. Gain jitter (scale spectrogram)
        if random.random() < 0.6:
            gain_db = random.uniform(-12, 6)
            result = result + gain_db

        # 3. Bandpass filter (attenuate high/low freq bands)
        if random.random() < 0.4:
            n_mels = result.shape[0]
            low_cut = random.randint(5, 20)
            high_cut = random.randint(n_mels - 25, n_mels - 5)
            fade = np.ones(n_mels, dtype=np.float32)
            fade[:low_cut] = np.linspace(0.1, 1.0, low_cut)
            fade[high_cut:] = np.linspace(1.0, 0.1, n_mels - high_cut)
            result = result * fade[:, np.newaxis]

        # 4. Reverb (blur along time axis)
        if random.random() < 0.5:
            kernel_len = random.randint(3, 8)
            decay = np.exp(-np.arange(kernel_len) * random.uniform(0.3, 1.0))
            decay = decay / decay.sum()
            for i in range(result.shape[0]):
                result[i] = np.convolve(result[i], decay, mode="same")

        # 5. Codec simulation (downsample + upsample along time)
        if random.random() < 0.3:
            t = result.shape[1]
            factor = random.choice([2, 3, 4])
            down = scipy_signal.resample(result, t // factor, axis=1)
            result = scipy_signal.resample(down, t, axis=1)

        return result.astype(np.float32)


class FineTuneDataset(Dataset):
    """
    Dataset that augments existing mel spectrograms with fast mic-simulation.

    Pipeline per sample:
        1. Load .npy mel spectrogram (128 × T)
        2. Apply FastMicAugment (spectrogram-domain noise, gain, reverb, bandpass, codec)
        3. Apply Normalize + SpecAugment (as in original training)

    For validation: no FastMicAugment or SpecAugment.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        features_dir: str | Path,
        split: Literal["train", "val", "test"] = "train",
        augment_prob: float = 0.8,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.features_dir = Path(features_dir)
        self.split = split

        # Standard normalization (always applied)
        self.normalize = Normalize()

        # Fast spectrogram-domain mic augmentation (train only)
        if split == "train":
            self.mic_augment = FastMicAugment(p=augment_prob)
            self.spec_augment = SpecAugment()
        else:
            self.mic_augment = None
            self.spec_augment = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]

        # 1. Load original mel
        mel_path = self.features_dir / row["mel_path"]
        mel = np.load(str(mel_path))  # (128, T)

        # 2. Apply fast spectrogram-domain mic augmentation
        if self.mic_augment is not None:
            mel = self.mic_augment(mel)

        # 3. To tensor
        mel_tensor = torch.from_numpy(mel).float().unsqueeze(0)  # (1, 128, T)

        # 4. Normalize
        mel_tensor = self.normalize(mel_tensor)

        # 5. SpecAugment (train only)
        if self.spec_augment is not None:
            mel_tensor = self.spec_augment(mel_tensor)

        label = int(row["is_cough"])

        return mel_tensor, label

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
    """Split manifest by subject_id (same as original dataset.py)."""
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(manifest_path)
    df = df[~df["is_silent"]].copy()

    subjects = df["subject_id"].unique()
    train_subjects, valtest_subjects = train_test_split(
        subjects, train_size=train_ratio, random_state=seed
    )
    relative_val = val_ratio / (1.0 - train_ratio)
    val_subjects, test_subjects = train_test_split(
        valtest_subjects, train_size=relative_val, random_state=seed
    )

    return {
        "train": df[df["subject_id"].isin(train_subjects)].copy(),
        "val": df[df["subject_id"].isin(val_subjects)].copy(),
        "test": df[df["subject_id"].isin(test_subjects)].copy(),
    }


def create_finetune_loaders(
    manifest_path: str | Path = "datasets/features/manifest.csv",
    features_dir: str | Path = "datasets/features",
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
    augment_prob: float = 0.8,
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders with mic augmentation for fine-tuning.

    Args:
        manifest_path: path to manifest.csv
        features_dir: root dir containing mel/ subfolder
        batch_size: batch size
        num_workers: DataLoader workers
        seed: random seed
        augment_prob: probability of applying mic augmentation per sample
        pin_memory: pin memory for GPU

    Returns:
        Dict with "train", "val", "test" DataLoaders.
    """
    splits = split_by_subject(manifest_path, seed=seed)
    loaders: dict[str, DataLoader] = {}

    for split_name, df in splits.items():
        dataset = FineTuneDataset(
            df=df,
            features_dir=features_dir,
            split=split_name,  # type: ignore[arg-type]
            augment_prob=augment_prob,
        )

        sampler = None
        shuffle = split_name == "train"

        if split_name == "train":
            labels = df["is_cough"].astype(int).values
            class_counts = np.bincount(labels)
            weights = 1.0 / class_counts[labels]
            sampler = WeightedRandomSampler(
                weights=weights.tolist(),
                num_samples=len(labels),
                replacement=True,
            )
            shuffle = False

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
