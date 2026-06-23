"""
Tests for models.cough_detector.model

Verifies:
    - Forward pass produces correct output shapes
    - Embedding head output is L2-normalised
    - Checkpoint save/load round-trip preserves weights
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.cough_detector.model import CoughDetector
from models.shared.checkpoint import load_checkpoint, save_checkpoint


class TestCoughDetector:
    """Tests for the CoughDetector model."""

    @pytest.fixture()
    def model(self) -> CoughDetector:
        """Create a small model (no pretrained weights for speed)."""
        return CoughDetector(pretrained=False)

    @pytest.fixture()
    def dummy_input(self) -> torch.Tensor:
        """Dummy mel spectrogram batch: (B=2, C=1, n_mels=128, T=94)."""
        return torch.randn(2, 1, 128, 94)

    def test_forward_shapes(
        self, model: CoughDetector, dummy_input: torch.Tensor
    ) -> None:
        """Forward pass should produce (B, 1) logits and (B, 512) embeddings."""
        logits, embeddings = model(dummy_input)

        assert logits.shape == (2, 1)
        assert embeddings.shape == (2, 512)

    def test_embeddings_l2_normalised(
        self, model: CoughDetector, dummy_input: torch.Tensor
    ) -> None:
        """Embedding vectors should be L2-normalised (unit norm)."""
        _, embeddings = model(dummy_input)
        norms = embeddings.norm(dim=1)

        torch.testing.assert_close(norms, torch.ones(2), atol=1e-5, rtol=1e-5)

    def test_encode_shape(
        self, model: CoughDetector, dummy_input: torch.Tensor
    ) -> None:
        """Encoder should produce (B, 512) feature vectors."""
        features = model.encode(dummy_input)
        assert features.shape == (2, 512)

    def test_predict_shape(
        self, model: CoughDetector, dummy_input: torch.Tensor
    ) -> None:
        """predict() should return probabilities in [0, 1]."""
        probs = model.predict(dummy_input)
        assert probs.shape == (2, 1)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_variable_time_frames(self, model: CoughDetector) -> None:
        """Model should handle different time-frame lengths."""
        model.eval()  # BatchNorm requires batch > 1 in training mode
        for t_len in [50, 94, 200]:
            x = torch.randn(2, 1, 128, t_len)
            logits, embeddings = model(x)
            assert logits.shape == (2, 1)
            assert embeddings.shape == (2, 512)


class TestCheckpointRoundTrip:
    """Tests for save/load checkpoint."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """Saving and loading a checkpoint should preserve model weights."""
        model = CoughDetector(pretrained=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Run a dummy forward/backward to generate optimizer state
        dummy = torch.randn(2, 1, 128, 94)
        logits, _ = model(dummy)
        loss = logits.sum()
        loss.backward()
        optimizer.step()

        # Save
        path = tmp_path / "test_checkpoint.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=5,
            metrics={"loss": 0.5, "auc": 0.8},
            path=path,
        )

        assert path.exists()

        # Load into a fresh model
        model2 = CoughDetector(pretrained=False)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)

        info = load_checkpoint(path, model2, optimizer2)

        assert info["epoch"] == 5
        assert info["metrics"]["auc"] == 0.8

        # Weights should match
        for (n1, p1), (_n2, p2) in zip(
            model.named_parameters(), model2.named_parameters(), strict=False
        ):
            torch.testing.assert_close(p1, p2, msg=f"Mismatch in {n1}")
