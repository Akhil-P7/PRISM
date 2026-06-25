"""
PRISM Models — Shared SpecAugment Transforms

Implements SpecAugment-style augmentation as torch.nn.Module subclasses
for use in the CoughDataset training pipeline.

References:
    Park et al. (2019) — SpecAugment: A Simple Data Augmentation Method for ASR
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn


class TimeMask(nn.Module):
    """Randomly zero out a contiguous block of time frames."""

    def __init__(self, max_width: int = 30, num_masks: int = 1) -> None:
        super().__init__()
        self.max_width = max_width
        self.num_masks = num_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (C, n_mels, time_frames) or (n_mels, time_frames)
        """
        t_len = x.shape[-1]
        for _ in range(self.num_masks):
            width = random.randint(0, min(self.max_width, t_len))
            start = random.randint(0, max(0, t_len - width))
            x = x.clone()
            x[..., start : start + width] = 0.0
        return x


class FreqMask(nn.Module):
    """Randomly zero out a contiguous block of frequency bins (mel channels)."""

    def __init__(self, max_width: int = 20, num_masks: int = 1) -> None:
        super().__init__()
        self.max_width = max_width
        self.num_masks = num_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (C, n_mels, time_frames) or (n_mels, time_frames)
        """
        f_len = x.shape[-2]
        for _ in range(self.num_masks):
            width = random.randint(0, min(self.max_width, f_len))
            start = random.randint(0, max(0, f_len - width))
            x = x.clone()
            x[..., start : start + width, :] = 0.0
        return x


class GaussianNoise(nn.Module):
    """Add random Gaussian noise to the spectrogram."""

    def __init__(self, noise_factor: float = 0.005) -> None:
        super().__init__()
        self.noise_factor = noise_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.noise_factor
        return x + noise


class SpecAugment(nn.Module):
    """
    Combined SpecAugment transform: FreqMask + TimeMask + optional noise.

    Configured from training.yaml augmentation settings.
    Applied only during training (not val/test).
    """

    def __init__(
        self,
        time_mask: bool = True,
        freq_mask: bool = True,
        gaussian_noise: bool = True,
        noise_factor: float = 0.005,
        time_max_width: int = 30,
        freq_max_width: int = 20,
    ) -> None:
        super().__init__()
        transforms: list[nn.Module] = []
        if freq_mask:
            transforms.append(FreqMask(max_width=freq_max_width))
        if time_mask:
            transforms.append(TimeMask(max_width=time_max_width))
        if gaussian_noise:
            transforms.append(GaussianNoise(noise_factor=noise_factor))
        self.transforms = nn.Sequential(*transforms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transforms(x)


class Normalize(nn.Module):
    """Normalize a spectrogram to zero mean and unit variance (per sample)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean()
        std = x.std().clamp(min=1e-6)
        return (x - mean) / std
