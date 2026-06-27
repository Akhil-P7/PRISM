import torch
import torch.nn as nn
from torch.autograd import Function

# ──────────────────────────────────────────────────────────────────
# Canonical disease taxonomy — single source of truth
# ──────────────────────────────────────────────────────────────────

DISEASE_CLASSES: list[str] = [
    "Healthy",
    "COVID-19",
    "COPD",
    "Asthma",
    "Pneumonia",
    "URTI",
    "LRTI",
    "Bronchiectasis",
    "Bronchiolitis",
]
NUM_DISEASE_CLASSES: int = len(DISEASE_CLASSES)


class GradientReversalLayer(Function):
    """
    Gradient Reversal Layer for Domain Adversarial Training.
    During forward pass, it acts as an identity transform.
    During backward pass, it multiplies the gradient by a negative constant (-alpha).
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DomainDiscriminator(nn.Module):
    """
    Discriminator that tries to guess the domain (e.g., Smartphone vs Stethoscope)
    from the extracted embeddings. Used to enforce domain-invariant features.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),  # Binary classification for domain
        )

    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        # Apply gradient reversal before passing to discriminator
        x_reversed = GradientReversalLayer.apply(x, alpha)
        return self.net(x_reversed)


class DiseaseClassifierHead(nn.Module):
    """
    Multi-class disease classification head applied on top of the acoustic embeddings.
    """

    def __init__(
        self, input_dim: int = 512, hidden_dim: int = 256, num_classes: int = 9
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns logits (pre-softmax)
        return self.net(x)


class UnifiedUniversalClassifier(nn.Module):
    """
    A unified wrapper combining the Classifier Head and Domain Discriminator.
    Useful for encapsulating the forward pass during Domain Adversarial Training.
    """

    def __init__(self, input_dim: int = 512, num_classes: int = 9):
        super().__init__()
        self.classifier = DiseaseClassifierHead(
            input_dim=input_dim, num_classes=num_classes
        )
        self.domain_discriminator = DomainDiscriminator(input_dim=input_dim)

    def forward(
        self, x: torch.Tensor, alpha: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        disease_logits = self.classifier(x)
        domain_logits = self.domain_discriminator(x, alpha)
        return disease_logits, domain_logits
