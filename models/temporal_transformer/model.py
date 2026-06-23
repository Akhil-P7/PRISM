"""
PRISM Models — Temporal Transformer

Lightweight encoder-only Transformer for predicting disease trajectory
from a 30-day window of daily cough statistics.

Architecture::

    Input: (B, 30, 5)  ← 30 days × 5 features per day
           │
           ▼
    Linear Projection: (B, 30, 5) → (B, 30, d_model)
           │
           ▼
    + Sinusoidal Positional Encoding
           │
           ▼
    Transformer Encoder (n_layers, n_heads)
           │
           ▼
    Mean Pooling → (B, d_model)
           │
           ▼
    Classification Head → (B, num_classes)

Default hyperparameters (from configs/training.yaml):
    d_model=128, n_heads=4, n_layers=3, d_ff=256, dropout=0.1, num_classes=4
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Encodes position information using sine and cosine functions of
    different frequencies.  Does not require any learnable parameters.

    Args:
        d_model: dimension of the model embeddings
        max_len: maximum sequence length to pre-compute
        dropout: dropout rate applied after adding positional encoding
    """

    def __init__(
        self,
        d_model: int = 128,
        max_len: int = 100,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute positional encodings: (1, max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        # Register as buffer (not a parameter — no gradients)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (B, seq_len, d_model)

        Returns:
            x + positional encoding, with dropout applied
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """
    Temporal Transformer for disease trajectory classification.

    Takes a sequence of daily cough statistics and predicts one of
    4 trajectory classes: Stable (0), Improving (1), Increasing (2),
    Abnormal (3).

    Args:
        input_features: number of features per time step (default 5)
        d_model: transformer model dimension (default 128)
        n_heads: number of attention heads (default 4)
        n_layers: number of transformer encoder layers (default 3)
        d_ff: feed-forward hidden dimension (default 256)
        dropout: dropout rate (default 0.1)
        max_sequence_length: maximum sequence length (default 30)
        num_classes: number of output classes (default 4)
    """

    def __init__(
        self,
        input_features: int = 5,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_sequence_length: int = 30,
        num_classes: int = 4,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_classes = num_classes

        # --- Input projection: (B, seq, input_features) → (B, seq, d_model) ---
        self.input_projection = nn.Sequential(
            nn.Linear(input_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
        )

        # --- Positional encoding ---
        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_sequence_length + 10,  # small buffer
            dropout=dropout,
        )

        # --- Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # (B, seq, d_model) convention
            norm_first=True,  # Pre-LN (more stable training)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # --- Classification head ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: input tensor of shape (B, seq_len, input_features)
            padding_mask: optional boolean mask of shape (B, seq_len),
                          True for padded positions (to be ignored)

        Returns:
            logits: class logits of shape (B, num_classes)
        """
        # Project input features to model dimension
        x = self.input_projection(x)  # (B, seq, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)  # (B, seq, d_model)

        # Transformer encoder
        x = self.transformer_encoder(
            x,
            src_key_padding_mask=padding_mask,
        )  # (B, seq, d_model)

        # Mean pooling over sequence dimension
        # If padding_mask is provided, only pool over non-padded positions
        if padding_mask is not None:
            # padding_mask: True = padded, so we invert for weighting
            mask = ~padding_mask  # (B, seq)
            mask = mask.unsqueeze(-1).float()  # (B, seq, 1)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, d_model)
        else:
            x = x.mean(dim=1)  # (B, d_model)

        # Classification head
        logits = self.classifier(x)  # (B, num_classes)

        return logits

    def predict(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Convenience: returns predicted class probabilities (softmax)."""
        logits = self.forward(x, padding_mask)
        return torch.softmax(logits, dim=-1)

    def predict_class(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Convenience: returns predicted class indices."""
        logits = self.forward(x, padding_mask)
        return logits.argmax(dim=-1)


def build_temporal_model(
    config: dict | None = None,
    device: torch.device | None = None,
) -> TemporalTransformer:
    """
    Factory function to build the Temporal Transformer from config.

    Args:
        config: temporal_transformer section from training.yaml
        device: target device (default: auto-detect)

    Returns:
        TemporalTransformer model on the specified device.
    """
    if config is None:
        config = {}

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model = TemporalTransformer(
        input_features=config.get("input_features", 5),
        d_model=config.get("d_model", 128),
        n_heads=config.get("n_heads", 4),
        n_layers=config.get("n_layers", 3),
        d_ff=config.get("d_ff", 256),
        dropout=config.get("dropout", 0.1),
        max_sequence_length=config.get("max_sequence_length", 30),
        num_classes=config.get("num_classes", 4),
    )

    return model.to(device)
