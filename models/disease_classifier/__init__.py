"""
PRISM Models — Disease Classifier Module

Universal disease classifier with domain-adversarial training
for respiratory condition prediction from acoustic embeddings.
"""

from models.disease_classifier.classifier import (
    DISEASE_CLASSES,
    NUM_DISEASE_CLASSES,
    DiseaseClassifierHead,
    DomainDiscriminator,
    UnifiedUniversalClassifier,
)

__all__ = [
    "DISEASE_CLASSES",
    "NUM_DISEASE_CLASSES",
    "DiseaseClassifierHead",
    "DomainDiscriminator",
    "UnifiedUniversalClassifier",
]
