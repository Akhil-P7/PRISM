"""
PRISM Retrieval — Clinical Insight Generator

Template-based engine that converts a PatientMemory into a structured
ClinicalInsight document.  Phase 1 uses deterministic templates with
no LLM dependency — the output is fully reproducible and local.

Usage::

    from retrieval.insight_generator import generate_insight
    from retrieval.memory_builder import PatientMemory

    insight = generate_insight(memory)
    print(insight.summary)
    for obs in insight.observations:
        print(f"  [{obs.severity}] {obs.text}")
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from retrieval.insight_generator.templates import ALL_TEMPLATES
from retrieval.memory_builder.memory_builder import PatientMemory

# ──────────────────────────────────────────────────────────────────
# Output data classes
# ──────────────────────────────────────────────────────────────────


@dataclass
class Observation:
    """A single clinical observation within an insight."""

    category: str  # "trajectory", "nocturnal", "similar_cases", etc.
    severity: str  # "info", "low", "moderate", "high"
    text: str


@dataclass
class ClinicalInsight:
    """
    The final output of the RATM pipeline.

    Contains a summary narrative, individual observations,
    severity assessment, and metadata about how it was generated.
    """

    patient_id: str
    generated_at: str
    trajectory_class: str
    trajectory_confidence: float
    overall_severity: str  # "low", "moderate", "high"
    summary: str  # Primary narrative paragraph
    observations: list[Observation]
    similar_cases_count: int
    alerts_count: int
    templates_used: list[str]
    generation_method: str = "template_v1"
    disease_probabilities: dict[str, float] | None = None


# ──────────────────────────────────────────────────────────────────
# Severity computation
# ──────────────────────────────────────────────────────────────────

SEVERITY_RANK = {"info": 0, "low": 1, "moderate": 2, "high": 3}


def _compute_overall_severity(
    observations: list[Observation],
    trajectory_class: int,
) -> str:
    """
    Determine overall severity from observations and trajectory.

    Rules:
        - Any "high" observation → overall "high"
        - Trajectory Abnormal (3) → at least "moderate"
        - Trajectory Increasing (2) → at least "moderate"
        - Multiple "moderate" observations → "moderate"
        - Otherwise → "low"
    """
    max_obs_severity = max(
        (SEVERITY_RANK.get(o.severity, 0) for o in observations),
        default=0,
    )

    # Trajectory override
    if trajectory_class == 3 or trajectory_class == 2:  # Abnormal
        max_obs_severity = max(max_obs_severity, SEVERITY_RANK["moderate"])

    # Map back
    for name, rank in SEVERITY_RANK.items():
        if rank == max_obs_severity:
            return name

    return "low"


# ──────────────────────────────────────────────────────────────────
# Summary builder
# ──────────────────────────────────────────────────────────────────


def _build_summary(
    memory: PatientMemory,
    observations: list[Observation],
    severity: str,
) -> str:
    """
    Compose the primary summary paragraph from the most important observations.

    Takes the trajectory observation (always first) and appends
    the highest-severity non-trajectory observation for context.
    """
    # Start with trajectory observation
    traj_obs = [o for o in observations if o.category == "trajectory"]
    summary = traj_obs[0].text if traj_obs else ""

    # Add the most severe non-trajectory observation
    non_traj = [o for o in observations if o.category != "trajectory"]
    non_traj.sort(key=lambda o: SEVERITY_RANK.get(o.severity, 0), reverse=True)

    if non_traj:
        summary += " " + non_traj[0].text

    # Add similar cases reference if present
    sim_obs = [o for o in observations if o.category == "similar_cases"]
    if sim_obs and sim_obs[0] not in non_traj[:1]:
        summary += " " + sim_obs[0].text

    return summary


# ──────────────────────────────────────────────────────────────────
# Main generator
# ──────────────────────────────────────────────────────────────────


def generate_insight(memory: PatientMemory) -> ClinicalInsight:
    """
    Generate a ClinicalInsight from a PatientMemory.

    Evaluates all registered templates against the memory,
    collects applicable observations, computes severity,
    and assembles the final insight document.

    Args:
        memory: PatientMemory object from the Memory Builder.

    Returns:
        ClinicalInsight with narrative, observations, and metadata.
    """
    observations: list[Observation] = []
    templates_used: list[str] = []

    for template_fn in ALL_TEMPLATES:
        result = template_fn(memory)
        if result is not None:
            category, severity, text = result
            observations.append(
                Observation(
                    category=category,
                    severity=severity,
                    text=text,
                )
            )
            templates_used.append(template_fn.__name__)

    # Compute overall severity
    overall_severity = _compute_overall_severity(
        observations,
        memory.trajectory.predicted_class,
    )

    # Build summary
    summary = _build_summary(memory, observations, overall_severity)

    return ClinicalInsight(
        patient_id=memory.subject_id,
        generated_at=datetime.now(UTC).isoformat(),
        trajectory_class=memory.trajectory.class_name,
        trajectory_confidence=memory.trajectory.confidence,
        overall_severity=overall_severity,
        summary=summary,
        observations=observations,
        similar_cases_count=len(memory.similar_cases),
        alerts_count=len(memory.alerts),
        templates_used=templates_used,
    )
