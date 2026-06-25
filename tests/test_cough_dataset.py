"""
Tests for models.cough_detector.dataset

Verifies:
    - Dataset loads without error
    - Subject-based splits have no overlap
    - Split sizes are reasonable
    - Augmentation is applied only in train split
    - Correct output shapes and types
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from models.cough_detector.dataset import CoughDataset, split_by_subject


@pytest.fixture()
def dummy_manifest(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal manifest + dummy .npy files for testing."""
    features_dir = tmp_path / "features"
    mel_dir = features_dir / "mel"
    mel_dir.mkdir(parents=True)

    rows = []
    # Create 10 subjects, each with 2 segments
    for subj_idx in range(10):
        subject_id = f"subject_{subj_idx:03d}"
        recording_id = f"rec_{subj_idx:03d}"
        is_cough = subj_idx % 2 == 0  # alternating

        for seg_idx in range(2):
            mel_rel = f"mel/{recording_id}_seg{seg_idx:03d}.npy"
            mel_path = features_dir / mel_rel

            # Create a dummy mel spectrogram (128, 94)
            mel = np.random.randn(128, 94).astype(np.float32)
            np.save(str(mel_path), mel)

            rows.append(
                {
                    "recording_id": recording_id,
                    "subject_id": subject_id,
                    "dataset": "TEST",
                    "segment_idx": seg_idx,
                    "label": "Healthy" if not is_cough else "Cough",
                    "is_cough": is_cough,
                    "mel_path": mel_rel,
                    "mfcc_path": mel_rel,  # reuse for simplicity
                    "duration": 3.0,
                    "rms_energy": 0.1,
                    "zcr": 0.05,
                    "is_silent": False,
                }
            )

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    return manifest_path, features_dir


class TestSplitBySubject:
    """Tests for subject-based splitting."""

    def test_no_subject_overlap(self, dummy_manifest: tuple[Path, Path]) -> None:
        """No subject should appear in more than one split."""
        manifest_path, _ = dummy_manifest
        splits = split_by_subject(manifest_path)

        train_subjects = set(splits["train"]["subject_id"])
        val_subjects = set(splits["val"]["subject_id"])
        test_subjects = set(splits["test"]["subject_id"])

        assert train_subjects.isdisjoint(val_subjects)
        assert train_subjects.isdisjoint(test_subjects)
        assert val_subjects.isdisjoint(test_subjects)

    def test_all_subjects_covered(self, dummy_manifest: tuple[Path, Path]) -> None:
        """All non-silent subjects should appear in exactly one split."""
        manifest_path, _ = dummy_manifest
        splits = split_by_subject(manifest_path)

        all_split_subjects = set()
        for df in splits.values():
            all_split_subjects.update(df["subject_id"].unique())

        df = pd.read_csv(manifest_path)
        df = df[~df["is_silent"]]
        original_subjects = set(df["subject_id"].unique())

        assert all_split_subjects == original_subjects

    def test_splits_have_data(self, dummy_manifest: tuple[Path, Path]) -> None:
        """Each split should contain at least 1 row."""
        manifest_path, _ = dummy_manifest
        splits = split_by_subject(manifest_path)

        for split_name, df in splits.items():
            assert len(df) > 0, f"Split '{split_name}' is empty"


class TestCoughDataset:
    """Tests for the CoughDataset class."""

    def test_getitem_shapes(self, dummy_manifest: tuple[Path, Path]) -> None:
        """__getitem__ should return (mel, label) with correct shapes."""
        manifest_path, features_dir = dummy_manifest
        splits = split_by_subject(manifest_path)

        dataset = CoughDataset(
            df=splits["train"],
            features_dir=features_dir,
            split="train",
        )

        mel, label = dataset[0]

        assert isinstance(mel, torch.Tensor)
        assert mel.shape[0] == 1  # single channel
        assert mel.shape[1] == 128  # n_mels
        assert isinstance(label, int)
        assert label in (0, 1)

    def test_no_augment_in_val(self, dummy_manifest: tuple[Path, Path]) -> None:
        """Validation dataset should not have augmentation."""
        manifest_path, features_dir = dummy_manifest
        splits = split_by_subject(manifest_path)

        dataset = CoughDataset(
            df=splits["val"],
            features_dir=features_dir,
            split="val",
        )

        assert dataset.augment is None

    def test_augment_in_train(self, dummy_manifest: tuple[Path, Path]) -> None:
        """Training dataset should have augmentation enabled."""
        manifest_path, features_dir = dummy_manifest
        splits = split_by_subject(manifest_path)

        dataset = CoughDataset(
            df=splits["train"],
            features_dir=features_dir,
            split="train",
        )

        assert dataset.augment is not None

    def test_pos_weight(self, dummy_manifest: tuple[Path, Path]) -> None:
        """pos_weight should be a positive float."""
        manifest_path, features_dir = dummy_manifest
        splits = split_by_subject(manifest_path)

        dataset = CoughDataset(
            df=splits["train"],
            features_dir=features_dir,
            split="train",
        )

        assert dataset.pos_weight > 0
