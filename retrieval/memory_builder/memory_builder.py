"""
PRISM Retrieval — Patient Memory Builder

Constructs structured "Patient Memory" objects from raw data sources.
A PatientMemory is the single context document that feeds the Insight
Generator — it merges daily cough statistics, trajectory predictions,
similar-case retrieval results, and computed clinical alerts into one
coherent structure.

Usage::

    from retrieval.memory_builder import build_patient_memory

    memory = build_patient_memory(
        stats_df=daily_stats,          # DataFrame: 30 rows × 5 feature cols
        trajectory_result=traj,        # TrajectoryResult from Transformer
        similar_cases=cases,           # list[RetrievedCase] from retrieval
        subject_id="patient_001",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────

TRAJECTORY_NAMES = {0: "Stable", 1: "Improving", 2: "Increasing", 3: "Abnormal"}

FEATURE_COLUMNS = [
    "cough_count",
    "avg_duration",
    "avg_intensity",
    "night_ratio",
    "inter_cough_interval",
]


@dataclass
class CoughSummary:
    """Aggregated cough statistics across the monitoring window."""

    total_coughs: int
    avg_daily_count: float
    min_daily_count: int
    max_daily_count: int
    peak_day: int
    trend_slope: float  # linear regression slope (positive = increasing)
    trend_direction: str  # "increasing", "decreasing", "stable"
    night_cough_ratio: float
    avg_duration: float
    avg_intensity: float
    variability: float  # coefficient of variation of daily counts


@dataclass
class Alert:
    """A clinically relevant pattern detected in the data."""

    alert_type: (
        str  # "night_spike", "rapid_increase", "high_variability", "extreme_day"
    )
    description: str
    severity: str  # "low", "moderate", "high"
    day: int | None = None
    window: tuple[int, int] | None = None
    value: float | None = None


@dataclass
class TrajectoryResult:
    """Output from the Temporal Transformer."""

    predicted_class: int  # 0-3
    class_name: str
    confidence: float
    probabilities: list[float]


@dataclass
class SimilarCaseRef:
    """A reference to a similar historical case from retrieval."""

    subject_id: str
    similarity_score: float
    cough_ratio: float
    num_segments: int


@dataclass
class PatientMemory:
    """
    Complete structured context for a patient session.

    This is the single document that the Insight Generator consumes
    to produce clinical narratives.
    """

    subject_id: str
    generated_at: str
    window_days: int
    cough_summary: CoughSummary
    trajectory: TrajectoryResult
    similar_cases: list[SimilarCaseRef]
    alerts: list[Alert]


# ──────────────────────────────────────────────────────────────────
# Cough statistics aggregation
# ──────────────────────────────────────────────────────────────────


def summarize_cough_stats(stats_df: pd.DataFrame) -> CoughSummary:
    """
    Aggregate a 30-day daily statistics DataFrame into a CoughSummary.

    Args:
        stats_df: DataFrame with columns matching FEATURE_COLUMNS,
                  indexed or ordered by day (0-29).

    Returns:
        CoughSummary with computed aggregate metrics.
    """
    counts = stats_df["cough_count"].values.astype(float)
    n_days = len(counts)

    # Total and averages
    total = int(counts.sum())
    avg_daily = float(counts.mean())
    min_daily = int(counts.min())
    max_daily = int(counts.max())
    peak_day = int(np.argmax(counts))

    # Linear trend via least-squares
    days = np.arange(n_days, dtype=float)
    if n_days > 1 and counts.std() > 0:
        slope = float(np.polyfit(days, counts, 1)[0])
    else:
        slope = 0.0

    # Classify trend direction
    if abs(slope) < 0.1:
        direction = "stable"
    elif slope > 0:
        direction = "increasing"
    else:
        direction = "decreasing"

    # Night ratio (average across days)
    night_ratio = (
        float(stats_df["night_ratio"].mean())
        if "night_ratio" in stats_df.columns
        else 0.0
    )

    # Average duration and intensity
    avg_dur = (
        float(stats_df["avg_duration"].mean())
        if "avg_duration" in stats_df.columns
        else 0.0
    )
    avg_int = (
        float(stats_df["avg_intensity"].mean())
        if "avg_intensity" in stats_df.columns
        else 0.0
    )

    # Coefficient of variation (variability)
    cv = float(counts.std() / counts.mean()) if counts.mean() > 0 else 0.0

    return CoughSummary(
        total_coughs=total,
        avg_daily_count=round(avg_daily, 2),
        min_daily_count=min_daily,
        max_daily_count=max_daily,
        peak_day=peak_day,
        trend_slope=round(slope, 4),
        trend_direction=direction,
        night_cough_ratio=round(night_ratio, 4),
        avg_duration=round(avg_dur, 4),
        avg_intensity=round(avg_int, 4),
        variability=round(cv, 4),
    )


# ──────────────────────────────────────────────────────────────────
# Alert detection
# ──────────────────────────────────────────────────────────────────


def compute_alerts(stats_df: pd.DataFrame) -> list[Alert]:
    """
    Scan the 30-day statistics for clinically notable patterns.

    Detects:
        - Extreme single-day spikes (> 2 std above mean)
        - Elevated nocturnal coughing (night_ratio > 0.4)
        - Rapid increase windows (7-day slope > threshold)
        - High overall variability (CV > 0.5)

    Args:
        stats_df: DataFrame with daily cough statistics.

    Returns:
        List of Alert objects, sorted by severity (high first).
    """
    alerts: list[Alert] = []
    counts = stats_df["cough_count"].values.astype(float)
    n_days = len(counts)
    mean_count = counts.mean()
    std_count = counts.std() if n_days > 1 else 0.0

    # 1. Extreme single-day spikes
    spike_threshold = mean_count + 2 * std_count if std_count > 0 else mean_count * 2
    for day_idx in range(n_days):
        if counts[day_idx] > spike_threshold and counts[day_idx] > 15:
            alerts.append(
                Alert(
                    alert_type="extreme_day",
                    description=(
                        f"Day {day_idx}: {int(counts[day_idx])} coughs recorded "
                        f"(baseline avg: {mean_count:.1f}). This is significantly above normal."
                    ),
                    severity="high" if counts[day_idx] > mean_count * 3 else "moderate",
                    day=day_idx,
                    value=float(counts[day_idx]),
                )
            )

    # 2. Elevated nocturnal coughing
    if "night_ratio" in stats_df.columns:
        avg_night = float(stats_df["night_ratio"].mean())
        if avg_night > 0.4:
            alerts.append(
                Alert(
                    alert_type="night_spike",
                    description=(
                        f"Elevated nocturnal coughing detected: {avg_night:.0%} of coughs occur "
                        f"between 10 PM - 6 AM. This may indicate nocturnal asthma, GERD, "
                        f"or post-nasal drip."
                    ),
                    severity="moderate" if avg_night < 0.6 else "high",
                    value=avg_night,
                )
            )

    # 3. Rapid increase over a 7-day window
    if n_days >= 7:
        window_size = 7
        for start in range(n_days - window_size):
            window = counts[start : start + window_size]
            days_w = np.arange(window_size, dtype=float)
            slope = float(np.polyfit(days_w, window, 1)[0])
            if slope > 1.5:  # > 1.5 coughs/day increase per day
                pct_change = ((window[-1] - window[0]) / max(window[0], 1)) * 100
                alerts.append(
                    Alert(
                        alert_type="rapid_increase",
                        description=(
                            f"Rapid increase detected between days {start}-{start + window_size - 1}: "
                            f"cough frequency rose by {pct_change:.0f}% over this week."
                        ),
                        severity="high" if slope > 2.5 else "moderate",
                        window=(start, start + window_size - 1),
                        value=slope,
                    )
                )
                break  # report only the most significant window

    # 4. High overall variability
    if mean_count > 0:
        cv = std_count / mean_count
        if cv > 0.5:
            alerts.append(
                Alert(
                    alert_type="high_variability",
                    description=(
                        f"High day-to-day variability in cough frequency "
                        f"(coefficient of variation: {cv:.2f}). Irregular patterns "
                        f"may indicate an unstable condition."
                    ),
                    severity="moderate" if cv < 0.8 else "high",
                    value=cv,
                )
            )

    # Sort by severity: high > moderate > low
    severity_order = {"high": 0, "moderate": 1, "low": 2}
    alerts.sort(key=lambda a: severity_order.get(a.severity, 3))

    return alerts


# ──────────────────────────────────────────────────────────────────
# Main builder
# ──────────────────────────────────────────────────────────────────


def build_patient_memory(
    stats_df: pd.DataFrame,
    trajectory_result: TrajectoryResult,
    similar_cases: list[SimilarCaseRef] | None = None,
    subject_id: str = "unknown",
) -> PatientMemory:
    """
    Construct a complete PatientMemory from all data sources.

    Args:
        stats_df: 30-day daily cough statistics DataFrame.
        trajectory_result: prediction from the Temporal Transformer.
        similar_cases: retrieved similar cases (from retrieval engine).
        subject_id: patient identifier.

    Returns:
        PatientMemory object ready for the Insight Generator.
    """
    cough_summary = summarize_cough_stats(stats_df)
    alerts = compute_alerts(stats_df)

    return PatientMemory(
        subject_id=subject_id,
        generated_at=datetime.now(UTC).isoformat(),
        window_days=len(stats_df),
        cough_summary=cough_summary,
        trajectory=trajectory_result,
        similar_cases=similar_cases or [],
        alerts=alerts,
    )
