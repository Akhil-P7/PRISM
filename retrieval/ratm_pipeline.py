"""
PRISM Retrieval — RATM Pipeline (Retrieval-Augmented Temporal Modeling)

Top-level orchestrator that wires together:
    1. Temporal Transformer (trajectory prediction)
    2. TurboVec Retrieval Engine (similar case search)
    3. Memory Builder (structured context assembly)
    4. Insight Generator (clinical narrative generation)

Usage::

    from retrieval.ratm_pipeline import RATMPipeline

    pipeline = RATMPipeline()

    # From raw stats DataFrame
    insight = pipeline.generate_insight(
        patient_stats=stats_df,
        patient_embedding=embedding_512d,
        subject_id="patient_001",
    )

    # Quick demo with synthetic data
    insight = pipeline.generate_demo_insight()
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.temporal_transformer.model import TemporalTransformer
    from retrieval.retrieval_engine.retrieval_engine import RetrievalEngine

import numpy as np
import pandas as pd
import torch
from loguru import logger

from retrieval.insight_generator.insight_generator import (
    ClinicalInsight,
    generate_insight,
)
from retrieval.memory_builder.memory_builder import (
    SimilarCaseRef,
    TrajectoryResult,
    build_patient_memory,
)

TRAJECTORY_NAMES = {0: "Stable", 1: "Improving", 2: "Increasing", 3: "Abnormal"}


class RATMPipeline:
    """
    End-to-end RATM pipeline.

    Combines all PRISM AI components into a single
    ``generate_insight()`` call.
    """

    def __init__(
        self,
        transformer_checkpoint: str | Path | None = None,
        use_retrieval: bool = True,
    ) -> None:
        """
        Args:
            transformer_checkpoint: path to temporal_transformer_best.pt.
                If None, uses default from configs/inference.yaml.
            use_retrieval: whether to query TurboVec for similar cases.
                Set to False for environments without the index.
        """
        self._retrieval_engine: RetrievalEngine | None = None
        self._device: torch.device | None = None
        self._transformer: TemporalTransformer | None = None
        self._use_retrieval = use_retrieval

        # Resolve checkpoint path
        if transformer_checkpoint is None:
            transformer_checkpoint = "models/checkpoints/temporal_transformer_best.pt"
        self._transformer_checkpoint = Path(transformer_checkpoint)

    def _get_device(self) -> torch.device:
        """Get or detect the torch device."""
        if self._device is None:
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        return self._device

    def _get_transformer(self):
        """Lazy-load the Temporal Transformer model."""
        if self._transformer is not None:
            return self._transformer

        from models.shared.checkpoint import load_checkpoint
        from models.temporal_transformer.model import build_temporal_model

        device = self._get_device()
        model = build_temporal_model(device=device)

        if self._transformer_checkpoint.exists():
            load_checkpoint(str(self._transformer_checkpoint), model, device=device)
            logger.info(
                f"Loaded Temporal Transformer from {self._transformer_checkpoint}"
            )
        else:
            logger.warning(
                f"Transformer checkpoint not found: {self._transformer_checkpoint}. "
                "Using randomly initialized model."
            )

        model.eval()
        self._transformer = model
        return self._transformer

    def _get_retrieval_engine(self):
        """Lazy-load the retrieval engine."""
        if self._retrieval_engine is not None:
            return self._retrieval_engine

        from retrieval.retrieval_engine import RetrievalEngine

        self._retrieval_engine = RetrievalEngine()
        return self._retrieval_engine

    # ──────────────────────────────────────────────────────────────
    # Trajectory prediction
    # ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_trajectory(
        self,
        stats_df: pd.DataFrame,
    ) -> TrajectoryResult:
        """
        Run the Temporal Transformer on a 30-day stats DataFrame.

        Args:
            stats_df: DataFrame with 30 rows and the 5 feature columns.

        Returns:
            TrajectoryResult with predicted class, name, confidence, probabilities.
        """
        from models.temporal_transformer.dataset import FEATURE_COLUMNS

        model = self._get_transformer()
        device = self._get_device()

        # Extract feature matrix
        features = stats_df[FEATURE_COLUMNS].values.astype(np.float32)

        # Z-score normalize using training set statistics.
        # These are computed from the training CSV the model was trained on.
        # If the training CSV exists, compute from it; otherwise use stored defaults.
        feature_stats = self._get_feature_stats()
        for i, col in enumerate(FEATURE_COLUMNS):
            if col in feature_stats:
                mean, std = feature_stats[col]
                features[:, i] = (features[:, i] - mean) / std

        seq_tensor = torch.from_numpy(features).unsqueeze(0).to(device)  # (1, 30, 5)

        # Forward pass
        logits = model(seq_tensor)  # (1, 4)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])

        return TrajectoryResult(
            predicted_class=predicted_class,
            class_name=TRAJECTORY_NAMES[predicted_class],
            confidence=round(confidence, 4),
            probabilities=[round(float(p), 4) for p in probs],
        )

    def _get_feature_stats(self) -> dict[str, tuple[float, float]]:
        """Load or compute feature normalization stats from the training set."""
        if not hasattr(self, "_feature_stats_cache"):
            train_csv = Path("datasets/temporal/temporal_train.csv")
            if train_csv.exists():
                from models.temporal_transformer.dataset import TemporalDataset

                ds = TemporalDataset(str(train_csv), normalize=True)
                self._feature_stats_cache = ds.get_feature_stats()
            else:
                # Fallback: hardcoded from the training run (seed=42, 500/class)
                self._feature_stats_cache = {
                    "cough_count": (8.94, 5.89),
                    "avg_duration": (0.662, 0.200),
                    "avg_intensity": (0.508, 0.140),
                    "night_ratio": (0.325, 0.129),
                    "inter_cough_interval": (508.9, 378.6),
                }
        return self._feature_stats_cache

    # ──────────────────────────────────────────────────────────────
    # Similar case retrieval
    # ──────────────────────────────────────────────────────────────

    def retrieve_cases(
        self,
        patient_embedding: np.ndarray,
        predicted_trajectory: int,
        k: int = 5,
    ) -> list[SimilarCaseRef]:
        """
        Query the retrieval engine for similar historical cases.

        Args:
            patient_embedding: 512-dim CNN embedding.
            predicted_trajectory: predicted trajectory class (0-3).
            k: number of cases to retrieve.

        Returns:
            List of SimilarCaseRef for the Memory Builder.
        """
        if not self._use_retrieval:
            logger.info("Retrieval disabled — skipping similar case search.")
            return []

        try:
            engine = self._get_retrieval_engine()
            cases = engine.retrieve_similar_cases(
                query_embedding=patient_embedding,
                predicted_trajectory=predicted_trajectory,
                k=k,
            )
            return engine.to_similar_case_refs(cases)
        except Exception as e:
            logger.warning(f"Retrieval failed (non-fatal): {e}")
            return []

    # ──────────────────────────────────────────────────────────────
    # Full pipeline
    # ──────────────────────────────────────────────────────────────

    def generate_insight(
        self,
        patient_stats: pd.DataFrame,
        patient_embedding: np.ndarray | None = None,
        subject_id: str = "unknown",
        k: int = 5,
    ) -> ClinicalInsight:
        """
        Full RATM pipeline: predict trajectory, retrieve cases,
        build memory, generate insight.

        Args:
            patient_stats: 30-day daily stats DataFrame (30 rows × 5 features).
            patient_embedding: 512-dim CNN embedding for retrieval.
                               If None, retrieval is skipped.
            subject_id: patient identifier.
            k: number of similar cases to retrieve.

        Returns:
            ClinicalInsight with narrative, observations, and evidence.
        """
        logger.info(f"RATM pipeline started for subject={subject_id}")

        # Step 1: Predict trajectory
        trajectory = self.predict_trajectory(patient_stats)
        logger.info(
            f"  Trajectory: {trajectory.class_name} "
            f"(confidence={trajectory.confidence:.2%})"
        )

        # Step 2: Retrieve similar cases
        similar_cases: list[SimilarCaseRef] = []
        if patient_embedding is not None and self._use_retrieval:
            similar_cases = self.retrieve_cases(
                patient_embedding=patient_embedding,
                predicted_trajectory=trajectory.predicted_class,
                k=k,
            )
            logger.info(f"  Retrieved {len(similar_cases)} similar cases")
        else:
            logger.info("  Retrieval skipped (no embedding or disabled)")

        # Step 3: Build patient memory
        memory = build_patient_memory(
            stats_df=patient_stats,
            trajectory_result=trajectory,
            similar_cases=similar_cases,
            subject_id=subject_id,
        )
        logger.info(f"  Memory built: {len(memory.alerts)} alerts detected")

        # Step 4: Generate insight
        insight = generate_insight(memory)
        logger.info(
            f"  Insight generated: severity={insight.overall_severity}, "
            f"{len(insight.observations)} observations, "
            f"{len(insight.templates_used)} templates used"
        )

        return insight

    # ──────────────────────────────────────────────────────────────
    # Demo helper
    # ──────────────────────────────────────────────────────────────

    def generate_demo_insight(
        self,
        trajectory_class: int = 2,
        subject_id: str = "demo_patient_001",
    ) -> ClinicalInsight:
        """
        Generate a demo insight using synthetic data.

        Creates a synthetic 30-day patient matching the requested
        trajectory class and runs the full pipeline (without retrieval).

        Args:
            trajectory_class: desired trajectory (0=Stable, 1=Improving,
                              2=Increasing, 3=Abnormal).
            subject_id: demo patient ID.

        Returns:
            ClinicalInsight for the demo patient.
        """
        from models.temporal_transformer.generate_temporal_data import _GENERATORS

        logger.info(
            f"Generating demo insight: class={TRAJECTORY_NAMES[trajectory_class]}, "
            f"subject={subject_id}"
        )

        # Generate one synthetic patient
        rng = np.random.default_rng(42)
        generator_fn = _GENERATORS[trajectory_class]
        features = generator_fn(rng)  # (30, 5)

        # Build DataFrame
        from models.temporal_transformer.dataset import FEATURE_COLUMNS

        stats_df = pd.DataFrame(features, columns=FEATURE_COLUMNS)

        # Create some synthetic similar cases for demo purposes
        demo_cases = [
            SimilarCaseRef(
                subject_id=f"hist_{i:03d}",
                similarity_score=round(0.95 - i * 0.05, 2),
                cough_ratio=round(0.7 + np.random.default_rng(i).uniform(-0.2, 0.2), 2),
                num_segments=int(np.random.default_rng(i).integers(5, 20)),
            )
            for i in range(3)
        ]

        # Run pipeline without retrieval (use demo cases directly)
        trajectory = self.predict_trajectory(stats_df)

        memory = build_patient_memory(
            stats_df=stats_df,
            trajectory_result=trajectory,
            similar_cases=demo_cases,
            subject_id=subject_id,
        )

        return generate_insight(memory)
