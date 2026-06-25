"""
PRISM Pipelines — Feature Extractor

Converts a raw audio waveform (numpy array) into:
    1. Mel Spectrogram  — primary CNN input, shape (n_mels, T)
    2. MFCC             — secondary features, shape (n_mfcc, T)
    3. Audio statistics  — duration, RMS energy, zero-crossing rate

Handles segmentation: recordings longer than segment_duration are split
into overlapping segments.  Each segment produces a separate .npy file.

Usage:
    extractor = FeatureExtractor(config)
    segments = extractor.extract(waveform, recording_id="abc123")
    # Returns list of SegmentFeatures
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import librosa
import numpy as np

if TYPE_CHECKING:
    from pipelines.preprocessing.audio_config import AudioConfig


@dataclass
class SegmentFeatures:
    """Features for a single audio segment."""

    recording_id: str
    segment_idx: int
    mel_path: str  # relative path to saved .npy
    mfcc_path: str  # relative path to saved .npy
    duration: float  # segment duration in seconds
    rms_energy: float
    zero_crossing_rate: float
    is_silent: bool


class FeatureExtractor:
    """Extracts mel spectrograms and MFCCs from audio waveforms."""

    # Segments with RMS below this threshold are flagged as silent
    SILENCE_THRESHOLD = 1e-4

    def __init__(self, config: AudioConfig) -> None:
        self.config = config

    def _compute_mel(self, segment: np.ndarray) -> np.ndarray:
        """Compute log-mel spectrogram for a single segment."""
        mel = librosa.feature.melspectrogram(
            y=segment,
            sr=self.config.sample_rate,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            fmin=self.config.f_min,
            fmax=self.config.f_max,
        )
        # Convert to log scale (dB) for better CNN learning
        log_mel = librosa.power_to_db(mel, ref=np.max)
        return log_mel

    def _compute_mfcc(self, segment: np.ndarray) -> np.ndarray:
        """Compute MFCCs for a single segment."""
        mfcc = librosa.feature.mfcc(
            y=segment,
            sr=self.config.sample_rate,
            n_mfcc=self.config.n_mfcc,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
        )
        return mfcc

    def _compute_stats(self, segment: np.ndarray) -> tuple[float, float, bool]:
        """Compute basic audio statistics for a segment: (rms, zcr, is_silent)."""
        rms = float(np.sqrt(np.mean(segment**2)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=segment)))
        return (
            round(rms, 6),
            round(zcr, 6),
            rms < self.SILENCE_THRESHOLD,
        )

    def _segment_waveform(self, waveform: np.ndarray) -> list[np.ndarray]:
        """
        Split a waveform into fixed-length overlapping segments.
        Short recordings (< segment_duration) are zero-padded to
        segment_samples length.
        """
        seg_len = self.config.segment_samples
        hop = self.config.hop_samples
        total = len(waveform)

        # If shorter than one segment, zero-pad
        if total <= seg_len:
            padded = np.zeros(seg_len, dtype=waveform.dtype)
            padded[:total] = waveform
            return [padded]

        segments = []
        start = 0
        while start + seg_len <= total:
            segments.append(waveform[start : start + seg_len])
            start += hop

        # Handle the tail: if there's leftover audio, pad the last segment
        if start < total:
            tail = np.zeros(seg_len, dtype=waveform.dtype)
            remaining = total - start
            tail[:remaining] = waveform[start:]
            segments.append(tail)

        return segments

    def extract(
        self,
        waveform: np.ndarray,
        recording_id: str,
        save: bool = True,
    ) -> list[SegmentFeatures]:
        """
        Extract features from a waveform.

        Args:
            waveform: 1-D numpy array at config.sample_rate
            recording_id: unique ID for naming output files
            save: if True, save .npy files to disk

        Returns:
            List of SegmentFeatures, one per segment.
        """
        segments = self._segment_waveform(waveform)
        results: list[SegmentFeatures] = []

        for idx, segment in enumerate(segments):
            # Compute features
            mel = self._compute_mel(segment)
            mfcc = self._compute_mfcc(segment)
            rms_energy, zcr, is_silent = self._compute_stats(segment)

            # Build file paths
            seg_name = f"{recording_id}_seg{idx:03d}"
            mel_rel = f"mel/{seg_name}.npy"
            mfcc_rel = f"mfcc/{seg_name}.npy"

            if save:
                mel_path = Path(self.config.features_dir) / mel_rel
                mfcc_path = Path(self.config.features_dir) / mfcc_rel

                mel_path.parent.mkdir(parents=True, exist_ok=True)
                mfcc_path.parent.mkdir(parents=True, exist_ok=True)

                np.save(str(mel_path), mel)
                np.save(str(mfcc_path), mfcc)

            duration = len(segment) / self.config.sample_rate

            results.append(
                SegmentFeatures(
                    recording_id=recording_id,
                    segment_idx=idx,
                    mel_path=mel_rel,
                    mfcc_path=mfcc_rel,
                    duration=round(duration, 3),
                    rms_energy=rms_energy,
                    zero_crossing_rate=zcr,
                    is_silent=is_silent,
                )
            )

        return results
