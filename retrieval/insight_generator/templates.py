"""
PRISM Retrieval — Clinical Insight Templates

Template-based narrative generation system (Phase 1).

Each template is a function that takes a PatientMemory and returns
a formatted observation string (or None if the template doesn't apply).

Templates are grouped by category:
    - trajectory: overall trend narratives
    - nocturnal: night-coughing observations
    - alerts: clinically actionable warnings
    - similar_cases: evidence from retrieval
    - stability: reassuring observations for stable patients
"""

from __future__ import annotations

from retrieval.memory_builder.memory_builder import PatientMemory

# ──────────────────────────────────────────────────────────────────
# Template functions — each returns (category, severity, text) or None
# ──────────────────────────────────────────────────────────────────


def trajectory_summary(memory: PatientMemory) -> tuple[str, str, str] | None:
    """Generate the primary trajectory narrative."""
    cs = memory.cough_summary
    traj = memory.trajectory

    direction_text = {
        "increasing": "an upward trend",
        "decreasing": "a downward trend (improvement)",
        "stable": "a stable, consistent pattern",
    }
    trend_desc = direction_text.get(cs.trend_direction, "an observed pattern")

    text = (
        f"Over the {memory.window_days}-day monitoring window, "
        f"cough frequency shows {trend_desc} with an average of "
        f"{cs.avg_daily_count:.1f} coughs per day "
        f"(range: {cs.min_daily_count}-{cs.max_daily_count}). "
        f"The Temporal Transformer classified this trajectory as "
        f'"{traj.class_name}" with {traj.confidence:.0%} confidence.'
    )

    severity = "info"
    if traj.predicted_class == 2:  # Increasing
        severity = "moderate"
    elif traj.predicted_class == 3:  # Abnormal
        severity = "high"

    return ("trajectory", severity, text)


def peak_day_observation(memory: PatientMemory) -> tuple[str, str, str] | None:
    """Highlight the peak cough day if significantly above average."""
    cs = memory.cough_summary
    if cs.max_daily_count <= cs.avg_daily_count * 1.5:
        return None  # not notable

    ratio = cs.max_daily_count / max(cs.avg_daily_count, 1)
    text = (
        f"Peak cough activity occurred on day {cs.peak_day} "
        f"with {cs.max_daily_count} coughs "
        f"({ratio:.1f}x the daily average). "
    )

    if cs.max_daily_count > 20:
        text += "This elevated count may warrant clinical attention."
        severity = "moderate"
    else:
        text += "This is a notable but not extreme fluctuation."
        severity = "low"

    return ("peak_activity", severity, text)


def nocturnal_observation(memory: PatientMemory) -> tuple[str, str, str] | None:
    """Generate nocturnal coughing observation."""
    nr = memory.cough_summary.night_cough_ratio
    if nr < 0.3:
        return None  # normal nocturnal ratio, not worth reporting

    if nr >= 0.5:
        text = (
            f"Significant nocturnal coughing detected: {nr:.0%} of all cough events "
            f"occurred between 10 PM and 6 AM. Elevated nocturnal coughing is "
            f"commonly associated with nocturnal asthma, gastroesophageal reflux "
            f"disease (GERD), or post-nasal drip. Clinical correlation is recommended."
        )
        severity = "high"
    else:
        text = (
            f"Moderately elevated nocturnal coughing observed: {nr:.0%} of cough events "
            f"occurred during nighttime hours (10 PM - 6 AM), above the typical "
            f"threshold of 30%. This may affect sleep quality."
        )
        severity = "moderate"

    return ("nocturnal", severity, text)


def improving_observation(memory: PatientMemory) -> tuple[str, str, str] | None:
    """Positive observation for improving trajectories."""
    if memory.trajectory.predicted_class != 1:
        return None

    cs = memory.cough_summary
    text = (
        f"Positive trend detected: cough frequency is declining "
        f"(slope: {cs.trend_slope:.2f} coughs/day reduction). "
        f"The overall trajectory suggests the patient's condition "
        f"is improving. Continue current management and monitor for sustained improvement."
    )

    return ("improvement", "low", text)


def stable_observation(memory: PatientMemory) -> tuple[str, str, str] | None:
    """Reassuring observation for stable trajectories."""
    if memory.trajectory.predicted_class != 0:
        return None

    cs = memory.cough_summary
    text = (
        f"Cough pattern has remained stable throughout the {memory.window_days}-day "
        f"monitoring period with low variability (CV = {cs.variability:.2f}). "
        f"The consistent baseline of ~{cs.avg_daily_count:.0f} coughs/day suggests "
        f"a controlled chronic condition. No acute changes detected."
    )

    return ("stability", "low", text)


def abnormal_observation(memory: PatientMemory) -> tuple[str, str, str] | None:
    """Warning for abnormal trajectories."""
    if memory.trajectory.predicted_class != 3:
        return None

    n_alerts = len([a for a in memory.alerts if a.alert_type == "extreme_day"])
    text = (
        f"Irregular spike pattern detected with {n_alerts} anomalous "
        f"high-count day(s). This pattern does not follow a consistent trend "
        f"and may indicate an intermittent condition, environmental triggers, "
        f"or an unstable respiratory state. Clinical review is recommended."
    )

    return ("abnormality", "high", text)


def similar_cases_observation(memory: PatientMemory) -> tuple[str, str, str] | None:
    """Generate observation about retrieved similar cases."""
    cases = memory.similar_cases
    if not cases:
        return None

    n = len(cases)
    best_score = cases[0].similarity_score if cases else 0.0
    avg_score = sum(c.similarity_score for c in cases) / n if n > 0 else 0.0

    if n == 1:
        text = (
            f"1 historical case with a similar acoustic profile was identified "
            f"(similarity: {best_score:.2f}). "
        )
    else:
        scores_str = ", ".join(f"{c.similarity_score:.2f}" for c in cases[:3])
        text = (
            f"{n} historical cases with similar acoustic profiles were identified "
            f"(similarity scores: {scores_str}). "
            f"Average similarity: {avg_score:.2f}. "
        )

    # Add cough ratio context
    avg_cr = sum(c.cough_ratio for c in cases) / n if n > 0 else 0.0
    if avg_cr > 0.7:
        text += (
            "These matched cases predominantly contained cough events, "
            "suggesting the current patient's cough characteristics are "
            "consistent with known cough patterns in the database."
        )
    else:
        text += (
            "The matched cases contain a mix of cough and non-cough segments, "
            "indicating the overall acoustic profile (not just cough presence) "
            "is similar."
        )

    return ("similar_cases", "info", text)


def variability_observation(memory: PatientMemory) -> tuple[str, str, str] | None:
    """Note high day-to-day variability."""
    cv = memory.cough_summary.variability
    if cv < 0.4:
        return None

    if cv > 0.7:
        text = (
            f"High day-to-day variability in cough frequency observed "
            f"(coefficient of variation: {cv:.2f}). Irregular patterns "
            f"may suggest environmental triggers, inconsistent medication "
            f"adherence, or a condition with episodic exacerbations."
        )
        severity = "moderate"
    else:
        text = (
            f"Moderate variability in daily cough counts detected "
            f"(CV: {cv:.2f}). Some fluctuation is normal, but sustained "
            f"variability may warrant further investigation."
        )
        severity = "low"

    return ("variability", severity, text)


def intensity_observation(memory: PatientMemory) -> tuple[str, str, str] | None:
    """Note if average cough intensity is unusually high."""
    intensity = memory.cough_summary.avg_intensity
    if intensity < 0.65:
        return None

    text = (
        f"Cough intensity is elevated (average RMS energy: {intensity:.2f}, "
        f"where typical range is 0.3-0.6). Higher intensity coughs may indicate "
        f"more forceful respiratory episodes, potentially associated with "
        f"airway irritation or infection."
    )

    return ("intensity", "moderate", text)


# ──────────────────────────────────────────────────────────────────
# Template registry
# ──────────────────────────────────────────────────────────────────

# All templates in evaluation order.
# The generator calls each one; those returning None are skipped.
ALL_TEMPLATES = [
    trajectory_summary,
    peak_day_observation,
    nocturnal_observation,
    improving_observation,
    stable_observation,
    abnormal_observation,
    similar_cases_observation,
    variability_observation,
    intensity_observation,
]
