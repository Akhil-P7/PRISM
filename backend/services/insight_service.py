"""
PRISM Backend — Insight Service

Service layer that wraps the RATM pipeline for the FastAPI endpoints.
Provides singleton pipeline management and error handling.
"""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd
from loguru import logger

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

        return pipeline.generate_insight(
            patient_stats=stats_df,
            patient_embedding=embedding,
            subject_id=subject_id,
            k=k,
        )

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
