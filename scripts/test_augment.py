"""Unit test for waveform augmentation transforms."""

import typing

import numpy as np

from models.shared.waveform_augment import (
    BackgroundNoise,
    BandpassFilter,
    CodecSimulate,
    GainJitter,
    HardClip,
    MicAugment,
    RoomReverb,
)

rng = np.random.default_rng(42)
waveform = rng.standard_normal(48000).astype(np.float32) * 0.5


transforms: list[tuple[str, typing.Any]] = [
    ("RoomReverb", RoomReverb()),
    ("BackgroundNoise", BackgroundNoise()),
    ("GainJitter", GainJitter()),
    ("HardClip", HardClip()),
    ("CodecSimulate", CodecSimulate()),
    ("BandpassFilter", BandpassFilter()),
    ("MicAugment", MicAugment(p=1.0)),
]

all_ok = True
for name, t in transforms:
    result = t(waveform)
    ok = (
        result.shape == waveform.shape
        and not np.any(np.isnan(result))
        and result.dtype == np.float32
    )
    if not ok:
        all_ok = False
    status = "OK" if ok else "FAIL"
    print(
        f"{name:20s} shape={result.shape} nan={np.any(np.isnan(result))} range=[{result.min():.3f}, {result.max():.3f}] {status}",
        flush=True,
    )

print(f'\nAll transforms: {"PASSED" if all_ok else "FAILED"}', flush=True)
