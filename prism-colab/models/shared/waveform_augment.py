"""
PRISM Models — Waveform-Domain Augmentation for Mic Robustness

Transforms that simulate browser-microphone recording conditions
by operating on raw audio waveforms *before* mel spectrogram computation.

Transforms:
    RoomReverb       — synthetic room impulse response convolution
    BackgroundNoise  — additive colored noise at random SNR
    GainJitter       — random amplitude scaling (±12 dB)
    HardClip         — simulate mic overload / clipping
    CodecSimulate    — downsample + upsample to mimic compression
    BandpassFilter   — simulate cheap mic frequency response
    MicAugment       — composite chain applying random subsets

Usage::

    from models.shared.waveform_augment import MicAugment

    augment = MicAugment(sample_rate=16000, p=0.8)
    augmented_waveform = augment(waveform)  # numpy 1-D array
"""

from __future__ import annotations

import random

import numpy as np
from scipy import signal as scipy_signal

# ──────────────────────────────────────────────────────────────────
# Individual Transforms
# ──────────────────────────────────────────────────────────────────


class RoomReverb:
    """
    Convolve with a synthetic room impulse response (RIR).

    Generates a decaying noise burst that simulates early reflections
    and late reverberation in a small room (0.2–0.6s RT60).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        rt60_range: tuple[float, float] = (0.15, 0.5),
        wet_range: tuple[float, float] = (0.05, 0.3),
    ) -> None:
        self.sample_rate = sample_rate
        self.rt60_range = rt60_range
        self.wet_range = wet_range

    def _make_rir(self, rt60: float) -> np.ndarray:
        """Generate a simple synthetic room impulse response."""
        n_samples = int(rt60 * self.sample_rate)
        if n_samples < 2:
            return np.array([1.0])

        # Decaying noise burst
        t = np.arange(n_samples) / self.sample_rate
        decay = np.exp(-6.9 * t / rt60)  # -60 dB at RT60
        noise = np.random.randn(n_samples)
        rir = noise * decay

        # Normalize
        rir = rir / (np.abs(rir).max() + 1e-8)

        # Strong direct path
        rir[0] = 1.0

        return rir.astype(np.float32)

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        rt60 = random.uniform(*self.rt60_range)
        wet = random.uniform(*self.wet_range)

        rir = self._make_rir(rt60)

        # Convolve and trim to original length
        reverbed = scipy_signal.fftconvolve(waveform, rir, mode="full")[: len(waveform)]

        # Mix dry + wet
        result = (1.0 - wet) * waveform + wet * reverbed

        # Prevent clipping
        peak = np.abs(result).max()
        if peak > 1.0:
            result = result / peak

        return result.astype(np.float32)


class BackgroundNoise:
    """
    Add colored noise (white, pink, or brown) at a random SNR.

    Simulates ambient room noise, fan hum, and general mic noise.
    """

    def __init__(
        self,
        snr_range: tuple[float, float] = (5.0, 25.0),
        noise_types: list[str] | None = None,
    ) -> None:
        self.snr_range = snr_range
        self.noise_types = noise_types or ["white", "pink", "brown"]

    def _pink_noise(self, n: int) -> np.ndarray:
        """Generate pink noise (1/f spectrum)."""
        white = np.random.randn(n)
        # Apply 1/f filter in frequency domain
        freqs = np.fft.rfftfreq(n)
        freqs[0] = 1e-6  # avoid division by zero
        fft = np.fft.rfft(white)
        fft = fft / np.sqrt(freqs)
        pink = np.fft.irfft(fft, n=n)
        return pink.astype(np.float32)

    def _brown_noise(self, n: int) -> np.ndarray:
        """Generate brown noise (1/f^2 spectrum, cumulative sum of white)."""
        white = np.random.randn(n)
        brown = np.cumsum(white)
        brown = brown - brown.mean()
        return brown.astype(np.float32)

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        snr_db = random.uniform(*self.snr_range)
        noise_type = random.choice(self.noise_types)
        n = len(waveform)

        if noise_type == "white":
            noise = np.random.randn(n).astype(np.float32)
        elif noise_type == "pink":
            noise = self._pink_noise(n)
        else:  # brown
            noise = self._brown_noise(n)

        # Normalize noise to match signal level at desired SNR
        signal_power = np.mean(waveform**2) + 1e-8
        noise_power = np.mean(noise**2) + 1e-8
        snr_linear = 10.0 ** (snr_db / 10.0)
        scale = np.sqrt(signal_power / (noise_power * snr_linear))

        result = waveform + scale * noise

        # Prevent clipping
        peak = np.abs(result).max()
        if peak > 1.0:
            result = result / peak

        return result.astype(np.float32)


class GainJitter:
    """
    Apply random gain change to simulate varying mic sensitivity.

    Range is specified in dB (e.g., ±12 dB).
    """

    def __init__(self, db_range: tuple[float, float] = (-12.0, 6.0)) -> None:
        self.db_range = db_range

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        gain_db = random.uniform(*self.db_range)
        gain_linear = 10.0 ** (gain_db / 20.0)
        result = waveform * gain_linear

        # Soft clip to prevent extreme values
        result = np.clip(result, -1.0, 1.0)

        return result.astype(np.float32)


class HardClip:
    """
    Simulate microphone overload by hard-clipping the waveform.

    Clips at a random threshold between 0.3 and 0.9 of peak amplitude.
    """

    def __init__(
        self,
        clip_range: tuple[float, float] = (0.3, 0.9),
    ) -> None:
        self.clip_range = clip_range

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        threshold = random.uniform(*self.clip_range)
        result = np.clip(waveform, -threshold, threshold)
        return result.astype(np.float32)


class CodecSimulate:
    """
    Simulate lossy codec artifacts by downsampling then upsampling.

    Mimics the quality loss from WebM/OGG/MP3 encoding used by browsers.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        target_rates: list[int] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.target_rates = target_rates or [4000, 6000, 8000, 11025]

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        target_sr = random.choice(self.target_rates)

        if target_sr >= self.sample_rate:
            return waveform

        n_orig = len(waveform)

        # Downsample
        n_down = int(n_orig * target_sr / self.sample_rate)
        if n_down < 2:
            return waveform

        downsampled = scipy_signal.resample(waveform, n_down)

        # Upsample back to original rate
        result = scipy_signal.resample(downsampled, n_orig)

        return result.astype(np.float32)


class BandpassFilter:
    """
    Apply a bandpass filter to simulate cheap mic frequency response.

    Laptop and phone mics often have a narrower effective bandwidth
    than research-grade microphones.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        low_range: tuple[int, int] = (100, 400),
        high_range: tuple[int, int] = (4000, 7000),
    ) -> None:
        self.sample_rate = sample_rate
        self.low_range = low_range
        self.high_range = high_range

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        low_hz = random.randint(*self.low_range)
        high_hz = random.randint(*self.high_range)

        nyquist = self.sample_rate / 2.0
        low = low_hz / nyquist
        high = min(high_hz / nyquist, 0.99)

        if low >= high:
            return waveform

        try:
            sos = scipy_signal.butter(4, [low, high], btype="band", output="sos")
            result = scipy_signal.sosfilt(sos, waveform)
            return result.astype(np.float32)
        except ValueError:
            return waveform


# ──────────────────────────────────────────────────────────────────
# Composite Augmentation Chain
# ──────────────────────────────────────────────────────────────────


class MicAugment:
    """
    Composite augmentation chain that randomly applies a subset of
    waveform-domain transforms to simulate browser-mic conditions.

    Each transform is independently applied with its own probability,
    and the overall chain has a master probability gate.

    Args:
        sample_rate: audio sample rate (default 16000)
        p: probability that augmentation is applied at all (default 0.8)
        transform_probs: per-transform probability dict (optional)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        p: float = 0.8,
        transform_probs: dict[str, float] | None = None,
    ) -> None:
        self.p = p

        # Default per-transform probabilities
        probs = transform_probs or {
            "reverb": 0.5,
            "noise": 0.7,
            "gain": 0.6,
            "clip": 0.2,
            "codec": 0.4,
            "bandpass": 0.3,
        }

        import typing

        self.transforms: list[tuple[str, float, typing.Any]] = [
            ("reverb", probs.get("reverb", 0.5), RoomReverb(sample_rate=sample_rate)),
            ("noise", probs.get("noise", 0.7), BackgroundNoise()),
            ("gain", probs.get("gain", 0.6), GainJitter()),
            ("clip", probs.get("clip", 0.2), HardClip()),
            ("codec", probs.get("codec", 0.4), CodecSimulate(sample_rate=sample_rate)),
            (
                "bandpass",
                probs.get("bandpass", 0.3),
                BandpassFilter(sample_rate=sample_rate),
            ),
        ]

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a waveform."""
        # Master gate
        if random.random() > self.p:
            return waveform

        result = waveform.copy()

        for _name, prob, transform in self.transforms:
            if random.random() < prob:
                result = transform(result)

        # Final safety: ensure no NaN and clip to valid range
        result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
        result = np.clip(result, -1.0, 1.0)

        return result.astype(np.float32)
