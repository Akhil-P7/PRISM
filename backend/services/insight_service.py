"""
PRISM Backend — Insight Service

Service layer that wraps the RATM pipeline for the FastAPI endpoints.
Provides singleton pipeline management and error handling.
"""

from __future__ import annotations

import os
import threading

import numpy as np
import pandas as pd
import torch
from loguru import logger

from models.disease_classifier.classifier import DISEASE_CLASSES, DiseaseClassifierHead
from retrieval.insight_generator.insight_generator import ClinicalInsight
from retrieval.ratm_pipeline import RATMPipeline

# ──────────────────────────────────────────────────────────────────
# Singleton RATM pipeline
# ──────────────────────────────────────────────────────────────────

_pipeline: RATMPipeline | None = None
_lock = threading.Lock()


class InsightServiceError(Exception):
    """Raised when the insight service encounters an error."""


def get_pipeline(use_retrieval: bool = True) -> RATMPipeline:
    """
    Get or initialise the singleton RATM pipeline.

    Thread-safe: only the first caller performs initialisation.
    """
    global _pipeline

    if _pipeline is not None:
        return _pipeline

    with _lock:
        if _pipeline is not None:
            return _pipeline

        _pipeline = RATMPipeline(use_retrieval=use_retrieval)
        logger.info("RATM pipeline initialised")
        return _pipeline


# ──────────────────────────────────────────────────────────────────
# Cached disease classifier singleton
# ──────────────────────────────────────────────────────────────────

_disease_classifier: DiseaseClassifierHead | None = None
_classifier_lock = threading.Lock()

_DISEASE_CKPT_PATH = os.path.join("models", "checkpoints", "disease_classifier_v1.pt")


def _get_disease_classifier() -> DiseaseClassifierHead | None:
    """
    Lazy-load and cache the disease classifier.

    Returns None if the checkpoint is not available.
    Thread-safe: only the first caller loads weights.
    """
    global _disease_classifier

    if _disease_classifier is not None:
        return _disease_classifier

    with _classifier_lock:
        if _disease_classifier is not None:
            return _disease_classifier

        classifier = DiseaseClassifierHead(
            input_dim=512, num_classes=len(DISEASE_CLASSES)
        )

        if os.path.exists(_DISEASE_CKPT_PATH):
            classifier.load_state_dict(
                torch.load(
                    _DISEASE_CKPT_PATH,
                    map_location="cpu",
                    weights_only=True,
                )
            )
            logger.info(f"Disease classifier loaded from {_DISEASE_CKPT_PATH}")
        else:
            logger.warning(
                f"Disease classifier checkpoint not found: {_DISEASE_CKPT_PATH}. "
                "Using randomly initialised weights."
            )

        classifier.eval()
        _disease_classifier = classifier
        return _disease_classifier


def _predict_disease_probabilities(
    patient_embedding: list[float],
) -> dict[str, float] | None:
    """
    Run the cached disease classifier on a 512-dim embedding.

    Returns a dict mapping disease name → probability, or None on failure.
    """
    classifier = _get_disease_classifier()
    if classifier is None:
        return None

    try:
        emb_tensor = torch.tensor(patient_embedding, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = classifier(emb_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().tolist()

        return {
            cls: round(p, 4) for cls, p in zip(DISEASE_CLASSES, probs, strict=False)
        }
    except Exception as e:
        logger.warning(f"Failed to compute disease probabilities: {e}")
        return None


# ──────────────────────────────────────────────────────────────────
# Service operations
# ──────────────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "cough_count",
    "avg_duration",
    "avg_intensity",
    "night_ratio",
    "inter_cough_interval",
]


def generate_insight(
    patient_stats: list[dict[str, float]],
    patient_embedding: list[float] | None = None,
    subject_id: str = "unknown",
    k: int = 5,
) -> ClinicalInsight:
    """
    Generate a clinical insight from patient data.

    Args:
        patient_stats: list of daily stat dicts (one per day).
        patient_embedding: optional 512-dim embedding for retrieval.
        subject_id: patient identifier.
        k: number of similar cases to retrieve.

    Returns:
        ClinicalInsight from the RATM pipeline.
    """
    try:
        pipeline = get_pipeline()

        # Convert stats to DataFrame
        stats_df = pd.DataFrame(patient_stats)

        # Validate columns
        missing = [c for c in FEATURE_COLUMNS if c not in stats_df.columns]
        if missing:
            raise InsightServiceError(
                f"Missing required columns in patient_stats: {missing}. "
                f"Required: {FEATURE_COLUMNS}"
            )

        # Convert embedding
        embedding = None
        if patient_embedding is not None:
            embedding = np.array(patient_embedding, dtype=np.float32)

        insight = pipeline.generate_insight(
            patient_stats=stats_df,
            patient_embedding=embedding,
            subject_id=subject_id,
            k=k,
        )

        # Predict disease probabilities from the embedding
        if patient_embedding is not None:
            insight.disease_probabilities = _predict_disease_probabilities(
                patient_embedding
            )

        return insight

    except InsightServiceError:
        raise
    except Exception as e:
        logger.error(f"Insight generation failed: {e}")
        raise InsightServiceError(f"Failed to generate insight: {e}") from e


def generate_demo_insight(
    trajectory_class: int = 2,
    subject_id: str = "demo_patient_001",
) -> ClinicalInsight:
    """
    Generate a demo insight using synthetic data.

    Args:
        trajectory_class: trajectory class for demo (0-3).
        subject_id: demo patient ID.

    Returns:
        ClinicalInsight for the demo patient.
    """
    try:
        pipeline = get_pipeline(use_retrieval=False)
        return pipeline.generate_demo_insight(
            trajectory_class=trajectory_class,
            subject_id=subject_id,
        )
    except Exception as e:
        logger.error(f"Demo insight generation failed: {e}")
        raise InsightServiceError(f"Failed to generate demo insight: {e}") from e
