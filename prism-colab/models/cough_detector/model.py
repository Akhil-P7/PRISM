"""
PRISM Models — Cough Detection CNN

ResNet-18 backbone adapted for single-channel mel spectrograms with two heads:
    1. Cough classifier — binary logit (is_cough)
    2. Embedding head  — 512-dim L2-normalised vector (for TurboVec)

Architecture::

    Input: (B, 1, 128, T)
    → ResNet-18 encoder (modified first conv for 1 channel)
    → AdaptiveAvgPool2d → (B, 512)
    → [Head A] fc_cough → (B, 1)     # binary logit
    → [Head B] fc_embed → (B, 512)   # L2-normalised embedding
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torchvision.models import ResNet18_Weights, resnet18


class CoughDetector(nn.Module):
    """
    ResNet-18 based cough detector with dual output heads.

    Args:
        num_classes: number of output classes for the classifier head (default 1 for binary)
        embedding_dim: dimension of the embedding output (default 512)
        pretrained: whether to use ImageNet pretrained weights for the encoder
        dropout: dropout rate before the heads
    """

    def __init__(
        self,
        num_classes: int = 1,
        embedding_dim: int = 512,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Load ResNet-18 backbone
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        # Adapt first conv layer: 3 channels → 1 channel (mono mel spectrogram)
        original_conv = backbone.conv1
        self.conv1 = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        # If pretrained, average the 3-channel weights → 1 channel
        if pretrained:
            with torch.no_grad():
                self.conv1.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))

        # Use the rest of ResNet-18 (bn1 → layer4), skip avgpool and fc
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Global average pooling → (B, 512)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Feature dimension from ResNet-18
        feat_dim = 512

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

        # Head A: binary cough classifier
        self.fc_cough = nn.Linear(feat_dim, num_classes)

        # Head B: embedding projection
        self.fc_embed = nn.Sequential(
            nn.Linear(feat_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the 512-dim feature vector from the backbone.

        Args:
            x: input tensor of shape (B, 1, n_mels, T)

        Returns:
            features: tensor of shape (B, 512)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = x.flatten(1)  # (B, 512)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with both heads.

        Args:
            x: input tensor of shape (B, 1, n_mels, T)

        Returns:
            logits: raw cough classification logits, shape (B, num_classes)
            embeddings: L2-normalised embeddings, shape (B, embedding_dim)
        """
        features = self.encode(x)
        features = self.dropout(features)

        # Cough classification head
        logits = self.fc_cough(features)

        # Embedding head (L2 normalised)
        embeddings = self.fc_embed(features)
        embeddings = nnf.normalize(embeddings, p=2, dim=1)

        return logits, embeddings

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method: returns cough probability (sigmoid applied)."""
        logits, _ = self.forward(x)
        return torch.sigmoid(logits)


def build_model(
    num_classes: int = 1,
    embedding_dim: int = 512,
    pretrained: bool = True,
    device: torch.device | None = None,
) -> CoughDetector:
    """
    Factory function to build and place the model on the correct device.

    Args:
        num_classes: classifier output size
        embedding_dim: embedding vector size
        pretrained: use ImageNet pretrained backbone
        device: target device (default: auto-detect)

    Returns:
        CoughDetector model on the specified device.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model = CoughDetector(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        pretrained=pretrained,
    )
    return model.to(device)
