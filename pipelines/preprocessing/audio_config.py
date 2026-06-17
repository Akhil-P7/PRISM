"""
PRISM Pipelines — Audio Processing Configuration

Centralized dataclass that provides all audio and feature extraction
parameters. Reads defaults from configs/training.yaml but can be
overridden programmatically.

Used by:
    - AudioExtractor  (sample_rate for resampling)
    - FeatureExtractor (mel/mfcc parameters)
    - run_feature_extraction.py (segment_duration, output paths)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml  # type: ignore


@dataclass
class AudioConfig:
    """Immutable configuration for the audio feature extraction pipeline."""

    # ---- Resampling ----
    sample_rate: int = 16000
    channels: int = 1

    # ---- Segmentation ----
    segment_duration: float = 3.0  # seconds per segment
    overlap: float = 0.5  # seconds of overlap between segments

    # ---- Mel Spectrogram ----
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    f_min: int = 20
    f_max: int = 8000

    # ---- MFCC ----
    n_mfcc: int = 40

    # ---- Paths ----
    features_dir: str = field(
        default_factory=lambda: os.getenv("FEATURES_PATH", "./datasets/features")
    )
    raw_dir: str = "datasets/raw"

    # ---- ZIP archive paths (relative to project root) ----
    zip_paths: dict[str, str] = field(
        default_factory=lambda: {
            "coughvid": "datasets/raw/coughvid.zip",
            "coswara": "datasets/raw/coswara.zip",
            "icbhi": "datasets/raw/icbhi.zip",
        }
    )

    @property
    def mel_dir(self) -> Path:
        return Path(self.features_dir) / "mel"

    @property
    def mfcc_dir(self) -> Path:
        return Path(self.features_dir) / "mfcc"

    @property
    def manifest_path(self) -> Path:
        return Path(self.features_dir) / "manifest.csv"

    @property
    def segment_samples(self) -> int:
        """Number of audio samples per segment."""
        return int(self.segment_duration * self.sample_rate)

    @property
    def overlap_samples(self) -> int:
        """Number of overlapping audio samples between segments."""
        return int(self.overlap * self.sample_rate)

    @property
    def hop_samples(self) -> int:
        """Number of samples to advance between segments."""
        return self.segment_samples - self.overlap_samples

    @classmethod
    def from_yaml(cls, path: str = "configs/training.yaml") -> AudioConfig:
        """Load config from the training YAML file, falling back to defaults."""
        try:
            with open(path) as f:
                raw = yaml.safe_load(f)
        except FileNotFoundError:
            return cls()

        audio = raw.get("audio", {})
        spec = raw.get("spectrogram", {})

        return cls(
            sample_rate=audio.get("sample_rate", cls.sample_rate),
            channels=audio.get("channels", cls.channels),
            segment_duration=audio.get("segment_duration", cls.segment_duration),
            overlap=audio.get("overlap", cls.overlap),
            n_mels=spec.get("n_mels", cls.n_mels),
            n_fft=spec.get("n_fft", cls.n_fft),
            hop_length=spec.get("hop_length", cls.hop_length),
            f_min=spec.get("f_min", cls.f_min),
            f_max=spec.get("f_max", cls.f_max),
        )
