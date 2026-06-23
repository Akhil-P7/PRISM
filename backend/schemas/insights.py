"""
PRISM Backend — Insight API Schemas

Pydantic models for the clinical insight generation endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ──────────────────────────────────────────────────────────────────
# Request models
# ──────────────────────────────────────────────────────────────────


class InsightRequest(BaseModel):
    """Request body for insight generation."""

    patient_stats: list[dict[str, float]] = Field(
        ...,
        min_length=1,
        max_length=60,
        description=(
            "List of daily cough statistics (one dict per day). "
            "Each dict must have keys: cough_count, avg_duration, "
            "avg_intensity, night_ratio, inter_cough_interval."
        ),
    )
    patient_embedding: list[float] | None = Field(
        default=None,
        min_length=512,
        max_length=512,
        description=(
            "Optional 512-dim CNN embedding for similar case retrieval. "
            "If omitted, retrieval is skipped."
        ),
    )
    subject_id: str = Field(
        default="unknown",
        description="Patient subject identifier.",
    )
    k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of similar cases to retrieve.",
    )


class DemoInsightRequest(BaseModel):
    """Request body for demo insight generation."""

    trajectory_class: int = Field(
        default=2,
        ge=0,
        le=3,
        description=(
            "Desired trajectory class for the demo. "
            "0=Stable, 1=Improving, 2=Increasing, 3=Abnormal."
        ),
    )
    subject_id: str = Field(
        default="demo_patient_001",
        description="Demo patient identifier.",
    )


# ──────────────────────────────────────────────────────────────────
# Response models
# ──────────────────────────────────────────────────────────────────


class ObservationItem(BaseModel):
    """A single clinical observation."""

    category: str = Field(
        description="Observation category (trajectory, nocturnal, etc.)."
    )
    severity: str = Field(description="Severity level: info, low, moderate, high.")
    text: str = Field(description="Human-readable observation text.")


class AlertItem(BaseModel):
    """A detected clinical alert."""

    alert_type: str = Field(description="Type of alert.")
    description: str = Field(description="Alert description.")
    severity: str = Field(description="Severity: low, moderate, high.")
    day: int | None = Field(default=None, description="Day index if applicable.")
    value: float | None = Field(default=None, description="Associated metric value.")


class SimilarCaseItem(BaseModel):
    """A similar historical case reference."""

    subject_id: str = Field(description="Historical patient ID.")
    similarity_score: float = Field(description="Relevance score (0-1).")
    cough_ratio: float = Field(description="Fraction of segments that are coughs.")
    num_segments: int = Field(description="Number of matching segments.")


class InsightResponse(BaseModel):
    """Full clinical insight response."""

    patient_id: str = Field(description="Patient subject ID.")
    generated_at: str = Field(description="ISO timestamp of generation.")
    trajectory_class: str = Field(description="Predicted trajectory class name.")
    trajectory_confidence: float = Field(
        description="Confidence of the trajectory prediction."
    )
    overall_severity: str = Field(description="Overall severity assessment.")
    summary: str = Field(description="Primary narrative summary paragraph.")
    observations: list[ObservationItem] = Field(
        default_factory=list,
        description="Individual clinical observations.",
    )
    similar_cases: list[SimilarCaseItem] = Field(
        default_factory=list,
        description="Retrieved similar historical cases.",
    )
    alerts: list[AlertItem] = Field(
        default_factory=list,
        description="Detected clinical alerts.",
    )
    templates_used: list[str] = Field(
        default_factory=list,
        description="Template functions that contributed to this insight.",
    )
    generation_method: str = Field(
        default="template_v1",
        description="Generation method identifier.",
    )
