"""
PRISM — Disease Classification Training Pipeline

This script demonstrates the training loop for the Unified Universal Classifier
using Domain Adversarial Training. It includes a mock Dataset to simulate the
512-D embeddings coming from the CNN/Temporal models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from models.disease_classifier.classifier import (
    NUM_DISEASE_CLASSES,
    UnifiedUniversalClassifier,
)

NUM_CLASSES = NUM_DISEASE_CLASSES


class MockEmbeddingDataset(Dataset):
    """
    Mock dataset generating random 512-D embeddings to simulate
    the output of the PRISM feature extraction pipeline.
    """

    def __init__(self, num_samples: int = 1000):
        self.num_samples = num_samples
        # 512-D embeddings
        self.embeddings = torch.randn(num_samples, 512)
        # Random disease labels (0 to NUM_CLASSES - 1)
        self.labels = torch.randint(0, NUM_CLASSES, (num_samples,))
        # Domain labels: 0 for Smartphone (Coswara/COUGHVID), 1 for Stethoscope (ICBHI)
        self.domains = torch.randint(0, 2, (num_samples,)).float()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.domains[idx]


def train_epoch(
    model, dataloader, optimizer, criterion_disease, criterion_domain, epoch, alpha
):
    model.train()
    total_loss = 0.0
    correct_disease = 0
    correct_domain = 0
    total = 0

    for _batch_idx, (embeddings, labels, domains) in enumerate(dataloader):
        optimizer.zero_grad()

        # Forward pass
        # Alpha controls the gradient reversal weight
        disease_logits, domain_logits = model(embeddings, alpha=alpha)

        # Disease Loss (Cross Entropy)
        loss_disease = criterion_disease(disease_logits, labels)

        # Domain Loss (Binary Cross Entropy)
        # domain_logits is [batch_size, 1], domains is [batch_size]
        loss_domain = criterion_domain(domain_logits.squeeze(-1), domains)

        # Total Loss (minimize disease loss, maximize domain confusion via GRL)
        # Note: GRL already negates the gradient for the discriminator,
        # so we simply ADD the losses here.
        loss = loss_disease + loss_domain

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        preds = torch.argmax(disease_logits, dim=1)
        correct_disease += (preds == labels).sum().item()

        domain_preds = torch.round(torch.sigmoid(domain_logits.squeeze(-1)))
        correct_domain += (domain_preds == domains).sum().item()

        total += labels.size(0)

    acc_disease = correct_disease / total
    acc_domain = correct_domain / total
    logger.info(
        f"Epoch {epoch}: Loss={total_loss/len(dataloader):.4f} | Disease Acc={acc_disease:.4f} | Domain Acc={acc_domain:.4f}"
    )


def main():
    logger.info("Initializing PRISM Disease Classification Pipeline")

    # Initialize Dataset and DataLoader
    dataset = MockEmbeddingDataset(num_samples=2000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize Model
    model = UnifiedUniversalClassifier(input_dim=512, num_classes=NUM_CLASSES)

    # Loss functions
    criterion_disease = nn.CrossEntropyLoss()
    criterion_domain = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5

    logger.info("Starting Domain Adversarial Training")
    for epoch in range(1, epochs + 1):
        # Calculate alpha for GRL: gradually increases from 0 to 1
        # Formula: p = epoch / total_epochs, alpha = 2 / (1 + exp(-10 * p)) - 1
        p = float(epoch) / epochs
        alpha = 2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p)).item()) - 1.0

        train_epoch(
            model,
            dataloader,
            optimizer,
            criterion_disease,
            criterion_domain,
            epoch,
            alpha,
        )

    logger.info("Training complete. Model weights can be saved.")


if __name__ == "__main__":
    main()
