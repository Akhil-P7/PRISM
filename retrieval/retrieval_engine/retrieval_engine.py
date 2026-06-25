"""
PRISM Retrieval — Multi-Signal Retrieval Engine

Orchestrates retrieval by combining:
  1. TurboVec embedding similarity (acoustic fingerprint matching)
  2. Trajectory-aware boosting (prefer cases with matching trajectory class)

The engine sits between the raw TurboVec search (segment-level) and
the Memory Builder, providing filtered, ranked, and enriched results.

Usage::

    from retrieval.retrieval_engine import RetrievalEngine

    engine = RetrievalEngine()
    cases = engine.retrieve_similar_cases(
        query_embedding=embedding_512d,
        predicted_trajectory=2,  # Increasing
        k=5,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml  # type: ignore
from loguru import logger

from retrieval.memory_builder.memory_builder import SimilarCaseRef
from retrieval.vector_store.search import PatientMatch, TurboVecSearchEngine

# ──────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────


@dataclass
class RetrievedCase:
    """A fully enriched similar-case result from multi-signal retrieval."""

    subject_id: str
    embedding_similarity: float  # raw TurboVec cosine similarity
    relevance_score: float  # final score after boosting
    num_matching_segments: int
    cough_ratio: float
    trajectory_match: bool  # whether this case shares the predicted trajectory


# ──────────────────────────────────────────────────────────────────
# Retrieval Engine
# ──────────────────────────────────────────────────────────────────


class RetrievalEngine:
    """
    Multi-signal retrieval engine.

    Wraps TurboVec search and adds trajectory-aware ranking.
    """

    def __init__(
        self,
        search_engine: TurboVecSearchEngine | None = None,
        config_path: str | Path = "configs/retrieval.yaml",
    ) -> None:
        """
        Args:
            search_engine: pre-initialised TurboVecSearchEngine (optional).
                           If None, will lazy-load from config paths.
            config_path: path to retrieval.yaml for thresholds and settings.
        """
        self._search_engine = search_engine
        self._config = self._load_config(config_path)

        # Retrieval settings
        retrieval_cfg = self._config.get("retrieval", {})
        self.default_k = retrieval_cfg.get("top_k", 5)
        self.min_similarity = retrieval_cfg.get("min_similarity_score", 0.7)
        self.max_results = retrieval_cfg.get("max_results", 10)

        # Trajectory boost factor — cases with matching trajectory get a score boost
        self.trajectory_boost = 0.1

    @staticmethod
    def _load_config(config_path: str | Path) -> dict:
        """Load retrieval configuration."""
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _get_engine(self) -> TurboVecSearchEngine:
        """Lazy-load the TurboVec search engine."""
        if self._search_engine is not None:
            return self._search_engine

        # Import here to avoid circular deps and allow graceful degradation
        from backend.services.retrieval_service import get_engine

        self._search_engine = get_engine()
        return self._search_engine

    def retrieve_similar_cases(
        self,
        query_embedding: np.ndarray,
        predicted_trajectory: int | None = None,
        k: int | None = None,
        min_similarity: float | None = None,
    ) -> list[RetrievedCase]:
        """
        Find the most relevant historical cases for a patient.

        Steps:
            1. Query TurboVec for top-N similar patients (by embedding)
            2. Filter by minimum similarity threshold
            3. Boost cases with matching trajectory class
            4. Re-rank and return top-k

        Args:
            query_embedding: 512-dim L2-normalised embedding vector.
            predicted_trajectory: predicted trajectory class (0-3) for boosting.
            k: number of final results (default from config).
            min_similarity: minimum similarity threshold (default from config).

        Returns:
            List of RetrievedCase, sorted by relevance_score descending.
        """
        k = k or self.default_k
        min_sim = min_similarity if min_similarity is not None else self.min_similarity

        try:
            engine = self._get_engine()
        except Exception as e:
            logger.warning(
                f"TurboVec engine unavailable: {e}. Returning empty results."
            )
            return []

        # Step 1: Get patient-level matches from TurboVec
        # Fetch more than needed to allow for filtering
        fetch_k = min(k * 3, self.max_results)
        raw_patients: list[PatientMatch] = engine.search_by_subject(
            query=query_embedding,
            k=fetch_k,
        )

        # Step 2 & 3: Filter + boost + convert
        results: list[RetrievedCase] = []
        for pm in raw_patients:
            if pm.best_similarity < min_sim:
                continue

            # Base relevance = best similarity score
            relevance = pm.best_similarity

            # Trajectory boost (placeholder — in production, we'd look up the
            # historical patient's trajectory from a database. For now, we
            # use cough_ratio as a proxy: high cough_ratio patients are more
            # likely to be "Increasing/Abnormal" trajectories)
            trajectory_match = False
            if predicted_trajectory is not None and (
                (predicted_trajectory in (2, 3) and pm.cough_ratio > 0.6)
                or (predicted_trajectory in (0, 1) and pm.cough_ratio <= 0.6)
            ):
                trajectory_match = True
                relevance += self.trajectory_boost

            results.append(
                RetrievedCase(
                    subject_id=pm.subject_id,
                    embedding_similarity=round(pm.best_similarity, 4),
                    relevance_score=round(min(relevance, 1.0), 4),
                    num_matching_segments=pm.num_matching_segments,
                    cough_ratio=round(pm.cough_ratio, 4),
                    trajectory_match=trajectory_match,
                )
            )

        # Step 4: Sort by relevance and take top-k
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        final = results[:k]

        logger.info(
            f"Retrieval: {len(raw_patients)} raw matches -> "
            f"{len(results)} after filter (min_sim={min_sim}) -> "
            f"{len(final)} returned"
        )

        return final

    def to_similar_case_refs(self, cases: list[RetrievedCase]) -> list[SimilarCaseRef]:
        """Convert RetrievedCase list to SimilarCaseRef list for the Memory Builder."""
        return [
            SimilarCaseRef(
                subject_id=c.subject_id,
                similarity_score=c.relevance_score,
                cough_ratio=c.cough_ratio,
                num_segments=c.num_matching_segments,
            )
            for c in cases
        ]
