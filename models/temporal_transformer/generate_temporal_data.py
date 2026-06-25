"""
PRISM Models — Synthetic Temporal Data Generator

Generates realistic 30-day patient cough timelines for training the
Temporal Transformer.  Each synthetic patient has a known trajectory
label (Stable / Improving / Increasing / Abnormal) and 30 days of
5 daily features:

    cough_count        — number of cough events in a day
    avg_duration       — average cough duration (seconds)
    avg_intensity      — average RMS energy of cough events
    night_ratio        — fraction of coughs between 10 PM – 6 AM
    inter_cough_interval — average seconds between consecutive coughs

Usage::

    # Generate with defaults (2000 patients, seed 42)
    poetry run python -m models.temporal_transformer.generate_temporal_data

    # Custom count / seed
    poetry run python -m models.temporal_transformer.generate_temporal_data \
        --patients-per-class 500 --seed 42 --output-dir datasets/temporal
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table

console = Console()

# Class labels
TRAJECTORY_LABELS = {
    0: "Stable",
    1: "Improving",
    2: "Increasing",
    3: "Abnormal",
}

SEQUENCE_LENGTH = 30  # 30-day window


# ---------------------------------------------------------------------------
# Trajectory generators — one function per class
# ---------------------------------------------------------------------------


def _generate_stable(
    rng: np.random.Generator, n_days: int = SEQUENCE_LENGTH
) -> np.ndarray:
    """
    Stable trajectory: cough count hovers around a fixed baseline.

    Pattern: λ ≈ 8 ± noise, flat across 30 days.
    """
    base_lambda = rng.uniform(5, 12)  # per-patient baseline varies
    cough_counts = rng.poisson(lam=base_lambda, size=n_days).astype(float)

    # Duration: stable ~0.5-0.8s
    base_dur = rng.uniform(0.4, 0.9)
    avg_duration = rng.normal(loc=base_dur, scale=0.08, size=n_days).clip(0.2, 1.5)

    # Intensity: stable ~0.4-0.6
    base_int = rng.uniform(0.35, 0.65)
    avg_intensity = rng.normal(loc=base_int, scale=0.06, size=n_days).clip(0.1, 0.95)

    # Night ratio: stable ~0.2-0.4
    base_nr = rng.uniform(0.15, 0.40)
    night_ratio = rng.normal(loc=base_nr, scale=0.05, size=n_days).clip(0.0, 1.0)

    # Inter-cough interval: inversely related to count
    ici = np.where(cough_counts > 0, 3600 / (cough_counts + 1), 3600.0)
    ici += rng.normal(0, 50, size=n_days)
    ici = ici.clip(30, 3600)

    return np.stack(
        [cough_counts, avg_duration, avg_intensity, night_ratio, ici], axis=1
    )


def _generate_improving(
    rng: np.random.Generator, n_days: int = SEQUENCE_LENGTH
) -> np.ndarray:
    """
    Improving trajectory: cough count decreases over the 30-day window.

    Pattern: λ decreases from ~12-18 → 2-5 linearly, with Poisson noise.
    """
    start_lambda = rng.uniform(12, 18)
    end_lambda = rng.uniform(2, 5)
    lambdas = np.linspace(start_lambda, end_lambda, n_days)
    cough_counts = rng.poisson(lam=lambdas).astype(float)

    # Duration decreases slightly
    dur_start = rng.uniform(0.7, 1.2)
    dur_end = rng.uniform(0.3, 0.6)
    avg_duration = np.linspace(dur_start, dur_end, n_days)
    avg_duration += rng.normal(0, 0.07, size=n_days)
    avg_duration = avg_duration.clip(0.2, 1.5)

    # Intensity decreases
    int_start = rng.uniform(0.6, 0.85)
    int_end = rng.uniform(0.25, 0.45)
    avg_intensity = np.linspace(int_start, int_end, n_days)
    avg_intensity += rng.normal(0, 0.05, size=n_days)
    avg_intensity = avg_intensity.clip(0.1, 0.95)

    # Night ratio decreases
    nr_start = rng.uniform(0.4, 0.6)
    nr_end = rng.uniform(0.1, 0.25)
    night_ratio = np.linspace(nr_start, nr_end, n_days)
    night_ratio += rng.normal(0, 0.04, size=n_days)
    night_ratio = night_ratio.clip(0.0, 1.0)

    # ICI
    ici = np.where(cough_counts > 0, 3600 / (cough_counts + 1), 3600.0)
    ici += rng.normal(0, 50, size=n_days)
    ici = ici.clip(30, 3600)

    return np.stack(
        [cough_counts, avg_duration, avg_intensity, night_ratio, ici], axis=1
    )


def _generate_increasing(
    rng: np.random.Generator, n_days: int = SEQUENCE_LENGTH
) -> np.ndarray:
    """
    Increasing trajectory: cough count increases over the 30-day window.

    Pattern: λ increases from ~2-5 → 12-20 linearly, with Poisson noise.
    """
    start_lambda = rng.uniform(2, 5)
    end_lambda = rng.uniform(12, 20)
    lambdas = np.linspace(start_lambda, end_lambda, n_days)
    cough_counts = rng.poisson(lam=lambdas).astype(float)

    # Duration increases slightly
    dur_start = rng.uniform(0.3, 0.5)
    dur_end = rng.uniform(0.7, 1.2)
    avg_duration = np.linspace(dur_start, dur_end, n_days)
    avg_duration += rng.normal(0, 0.07, size=n_days)
    avg_duration = avg_duration.clip(0.2, 1.5)

    # Intensity increases
    int_start = rng.uniform(0.25, 0.4)
    int_end = rng.uniform(0.6, 0.85)
    avg_intensity = np.linspace(int_start, int_end, n_days)
    avg_intensity += rng.normal(0, 0.05, size=n_days)
    avg_intensity = avg_intensity.clip(0.1, 0.95)

    # Night ratio increases
    nr_start = rng.uniform(0.1, 0.25)
    nr_end = rng.uniform(0.4, 0.65)
    night_ratio = np.linspace(nr_start, nr_end, n_days)
    night_ratio += rng.normal(0, 0.04, size=n_days)
    night_ratio = night_ratio.clip(0.0, 1.0)

    # ICI
    ici = np.where(cough_counts > 0, 3600 / (cough_counts + 1), 3600.0)
    ici += rng.normal(0, 50, size=n_days)
    ici = ici.clip(30, 3600)

    return np.stack(
        [cough_counts, avg_duration, avg_intensity, night_ratio, ici], axis=1
    )


def _generate_abnormal(
    rng: np.random.Generator, n_days: int = SEQUENCE_LENGTH
) -> np.ndarray:
    """
    Abnormal trajectory: baseline with random large spikes.

    Pattern: baseline λ ≈ 5, with 3-6 random spike days where λ jumps to 20-35.
    """
    base_lambda = rng.uniform(3, 7)
    cough_counts = rng.poisson(lam=base_lambda, size=n_days).astype(float)

    # Inject spikes on random days
    n_spikes = rng.integers(3, 7)
    spike_days = rng.choice(n_days, size=n_spikes, replace=False)
    spike_magnitudes = rng.uniform(20, 35, size=n_spikes)
    cough_counts[spike_days] = rng.poisson(lam=spike_magnitudes)

    # Duration: irregular — spikes have longer coughs
    base_dur = rng.uniform(0.4, 0.7)
    avg_duration = rng.normal(loc=base_dur, scale=0.08, size=n_days)
    avg_duration[spike_days] = rng.normal(loc=1.1, scale=0.15, size=n_spikes)
    avg_duration = avg_duration.clip(0.2, 1.5)

    # Intensity: spikes are more intense
    base_int = rng.uniform(0.3, 0.5)
    avg_intensity = rng.normal(loc=base_int, scale=0.06, size=n_days)
    avg_intensity[spike_days] = rng.normal(loc=0.8, scale=0.08, size=n_spikes)
    avg_intensity = avg_intensity.clip(0.1, 0.95)

    # Night ratio: higher on spike days
    base_nr = rng.uniform(0.2, 0.35)
    night_ratio = rng.normal(loc=base_nr, scale=0.05, size=n_days)
    night_ratio[spike_days] = rng.normal(loc=0.7, scale=0.1, size=n_spikes)
    night_ratio = night_ratio.clip(0.0, 1.0)

    # ICI
    ici = np.where(cough_counts > 0, 3600 / (cough_counts + 1), 3600.0)
    ici += rng.normal(0, 50, size=n_days)
    ici = ici.clip(30, 3600)

    return np.stack(
        [cough_counts, avg_duration, avg_intensity, night_ratio, ici], axis=1
    )


# Map class ID → generator function
_GENERATORS = {
    0: _generate_stable,
    1: _generate_improving,
    2: _generate_increasing,
    3: _generate_abnormal,
}


# ---------------------------------------------------------------------------
# Main generation pipeline
# ---------------------------------------------------------------------------


def generate_dataset(
    patients_per_class: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a full synthetic temporal dataset.

    Args:
        patients_per_class: number of patients per trajectory class
        seed: random seed for reproducibility

    Returns:
        DataFrame with columns: patient_id, day, cough_count, avg_duration,
        avg_intensity, night_ratio, inter_cough_interval, trajectory_label
    """
    rng = np.random.default_rng(seed)

    all_rows: list[dict] = []
    patient_id = 0

    for class_id, generator_fn in _GENERATORS.items():
        for _ in range(patients_per_class):
            # Generate 30-day feature matrix (30, 5)
            features = generator_fn(rng)

            for day in range(SEQUENCE_LENGTH):
                all_rows.append(
                    {
                        "patient_id": patient_id,
                        "day": day,
                        "cough_count": features[day, 0],
                        "avg_duration": round(features[day, 1], 4),
                        "avg_intensity": round(features[day, 2], 4),
                        "night_ratio": round(features[day, 3], 4),
                        "inter_cough_interval": round(features[day, 4], 2),
                        "trajectory_label": class_id,
                        "trajectory_name": TRAJECTORY_LABELS[class_id],
                    }
                )
            patient_id += 1

    df = pd.DataFrame(all_rows)
    return df


def split_and_save(
    df: pd.DataFrame,
    output_dir: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, Path]:
    """
    Split by patient_id (no patient leakage) and save as CSV files.

    Returns:
        Dict mapping split name → file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    patient_ids = df["patient_id"].unique()
    rng.shuffle(patient_ids)

    n_total = len(patient_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train : n_train + n_val]
    test_ids = patient_ids[n_train + n_val :]

    splits = {
        "train": df[df["patient_id"].isin(train_ids)],
        "val": df[df["patient_id"].isin(val_ids)],
        "test": df[df["patient_id"].isin(test_ids)],
    }

    paths: dict[str, Path] = {}
    for split_name, split_df in splits.items():
        path = output_dir / f"temporal_{split_name}.csv"
        split_df.to_csv(path, index=False)
        paths[split_name] = path
        logger.info(
            f"  {split_name}: {len(split_df['patient_id'].unique())} patients, "
            f"{len(split_df)} rows → {path}"
        )

    return paths


def print_summary(df: pd.DataFrame) -> None:
    """Print a Rich table summarising the generated dataset."""
    table = Table(
        title="Synthetic Temporal Dataset Summary",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Trajectory", style="cyan")
    table.add_column("Patients", justify="right")
    table.add_column("Avg Cough/Day", justify="right")
    table.add_column("Avg Duration", justify="right")
    table.add_column("Avg Intensity", justify="right")
    table.add_column("Avg Night Ratio", justify="right")

    for class_id, class_name in TRAJECTORY_LABELS.items():
        subset = df[df["trajectory_label"] == class_id]
        n_patients = subset["patient_id"].nunique()
        avg_cough = subset["cough_count"].mean()
        avg_dur = subset["avg_duration"].mean()
        avg_int = subset["avg_intensity"].mean()
        avg_nr = subset["night_ratio"].mean()

        table.add_row(
            class_name,
            str(n_patients),
            f"{avg_cough:.1f}",
            f"{avg_dur:.3f}",
            f"{avg_int:.3f}",
            f"{avg_nr:.3f}",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PRISM — Generate Synthetic Temporal Dataset"
    )
    parser.add_argument(
        "--patients-per-class",
        type=int,
        default=500,
        help="Number of synthetic patients per trajectory class (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/temporal",
        help="Output directory for CSV files (default: datasets/temporal)",
    )

    args = parser.parse_args()

    console.print(
        "\n[bold cyan]PRISM — Synthetic Temporal Data Generator[/bold cyan]\n"
    )
    console.print(f"  Patients per class : {args.patients_per_class}")
    console.print(f"  Total patients     : {args.patients_per_class * 4}")
    console.print(f"  Sequence length    : {SEQUENCE_LENGTH} days")
    console.print("  Features per day   : 5")
    console.print(f"  Seed               : {args.seed}")
    console.print(f"  Output dir         : {args.output_dir}\n")

    # Generate
    console.print("[bold]Generating synthetic timelines...[/bold]")
    df = generate_dataset(
        patients_per_class=args.patients_per_class,
        seed=args.seed,
    )

    # Summary
    print_summary(df)

    # Split and save
    console.print("\n[bold]Splitting and saving...[/bold]")
    paths = split_and_save(df, output_dir=args.output_dir, seed=args.seed)

    console.print(
        "\n[green]OK - Synthetic temporal dataset generated successfully![/green]"
    )
    for split_name, path in paths.items():
        console.print(f"   {split_name}: {path}")


if __name__ == "__main__":
    main()
