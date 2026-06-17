"""
PRISM Pipelines — Audio Feature Extraction Runner

CLI entry point that orchestrates the full feature extraction pipeline:
    1. Queries PostgreSQL for recordings (with dataset + subject labels)
    2. Extracts audio from ZIP archives
    3. Computes mel spectrograms + MFCCs
    4. Saves .npy feature files to datasets/features/
    5. Writes manifest.csv for PyTorch DataLoader consumption

Usage:
    poetry run python -m pipelines.preprocessing.run_feature_extraction
    poetry run python -m pipelines.preprocessing.run_feature_extraction --dataset icbhi
    poetry run python -m pipelines.preprocessing.run_feature_extraction --dataset icbhi --limit 50
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from sqlalchemy import func

from database.connection import SessionLocal
from database.models.dataset import Dataset
from database.models.recording import Recording
from database.models.subject import Subject
from pipelines.preprocessing.audio_config import AudioConfig
from pipelines.preprocessing.audio_extractor import AudioExtractor
from pipelines.preprocessing.feature_extractor import FeatureExtractor

console = Console()


def query_recordings(
    session, dataset_name: str | None = None, limit: int | None = None
):
    """
    Query recordings from the database with their dataset and subject info.
    Returns list of dicts with keys: recording_id, file_path, dataset_name,
    subject_id, respiratory_condition, is_cough.
    """
    query = (
        session.query(
            Recording.id.label("recording_id"),
            Recording.file_path,
            Recording.is_cough,
            Dataset.name.label("dataset_name"),
            Subject.id.label("subject_id"),
            Subject.respiratory_condition,
        )
        .join(Subject, Recording.subject_id == Subject.id)
        .join(Dataset, Subject.dataset_id == Dataset.id)
    )

    if dataset_name:
        query = query.filter(func.upper(Dataset.name) == dataset_name.upper())

    if limit:
        query = query.limit(limit)

    rows = query.all()
    return [
        {
            "recording_id": str(row.recording_id),
            "file_path": row.file_path,
            "dataset_name": row.dataset_name,
            "subject_id": str(row.subject_id),
            "respiratory_condition": row.respiratory_condition or "Unknown",
            "is_cough": row.is_cough,
        }
        for row in rows
    ]


def run_feature_extraction(
    dataset_name: str | None = None,
    limit: int | None = None,
) -> dict:
    """
    Run the full feature extraction pipeline.

    Returns a summary dict with counts and timing.
    """
    config = AudioConfig.from_yaml()

    # Ensure output directories exist
    config.mel_dir.mkdir(parents=True, exist_ok=True)
    config.mfcc_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold cyan]PRISM Audio Feature Extraction Pipeline[/bold cyan]\n")
    console.print(f"  Sample Rate   : {config.sample_rate} Hz")
    console.print(
        f"  Segment Length: {config.segment_duration}s (overlap {config.overlap}s)"
    )
    console.print(f"  Mel Bins      : {config.n_mels}")
    console.print(f"  Output Dir    : {config.features_dir}")
    console.print()

    # Step 1: Query recordings
    session = SessionLocal()
    try:
        recordings = query_recordings(session, dataset_name, limit)
    finally:
        session.close()

    if not recordings:
        console.print("[yellow]No recordings found for the given filter.[/yellow]")
        return {"total": 0, "processed": 0, "failed": 0, "segments": 0}

    console.print(f"  Recordings    : {len(recordings)}")
    console.print()

    # Step 2: Extract and compute features
    audio_extractor = AudioExtractor(config)
    feature_extractor = FeatureExtractor(config)

    manifest_rows: list[dict] = []
    processed = 0
    failed = 0
    skipped = 0
    skipped_silent = 0
    total_segments = 0
    skip_existing = True  # skip recordings whose features already exist
    start_time = time.time()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    with progress, audio_extractor:
        task = progress.add_task(
            "Extracting features...",
            total=len(recordings),
        )

        for rec in recordings:
            recording_id = rec["recording_id"]
            file_path = rec["file_path"]
            dataset = rec["dataset_name"]

            # Skip if features already exist on disk
            if skip_existing:
                first_seg = (
                    Path(config.features_dir) / "mel" / f"{recording_id}_seg000.npy"
                )
                if first_seg.exists():
                    # Reconstruct manifest rows from existing files
                    seg_idx = 0
                    while True:
                        mel_rel = f"mel/{recording_id}_seg{seg_idx:03d}.npy"
                        mfcc_rel = f"mfcc/{recording_id}_seg{seg_idx:03d}.npy"
                        mel_file = Path(config.features_dir) / mel_rel
                        if not mel_file.exists():
                            break
                        manifest_rows.append(
                            {
                                "recording_id": recording_id,
                                "subject_id": rec["subject_id"],
                                "dataset": dataset,
                                "segment_idx": seg_idx,
                                "label": rec["respiratory_condition"],
                                "is_cough": rec["is_cough"],
                                "mel_path": mel_rel,
                                "mfcc_path": mfcc_rel,
                                "duration": config.segment_duration,
                                "rms_energy": -1,  # not recomputed
                                "zcr": -1,
                                "is_silent": False,
                            }
                        )
                        seg_idx += 1
                    total_segments += seg_idx
                    skipped += 1
                    progress.advance(task)
                    continue

            # Load audio
            waveform = audio_extractor.load(file_path, dataset)

            if waveform is None:
                failed += 1
                progress.advance(task)
                continue

            # Extract features (segments)
            segments = feature_extractor.extract(
                waveform,
                recording_id=recording_id,
                save=True,
            )

            for seg in segments:
                if seg.is_silent:
                    skipped_silent += 1

                manifest_rows.append(
                    {
                        "recording_id": recording_id,
                        "subject_id": rec["subject_id"],
                        "dataset": dataset,
                        "segment_idx": seg.segment_idx,
                        "label": rec["respiratory_condition"],
                        "is_cough": rec["is_cough"],
                        "mel_path": seg.mel_path,
                        "mfcc_path": seg.mfcc_path,
                        "duration": seg.duration,
                        "rms_energy": seg.rms_energy,
                        "zcr": seg.zero_crossing_rate,
                        "is_silent": seg.is_silent,
                    }
                )

            processed += 1
            total_segments += len(segments)
            progress.advance(task)

    elapsed = time.time() - start_time

    # Step 3: Write manifest.csv
    if manifest_rows:
        manifest_path = config.manifest_path
        fieldnames = list(manifest_rows[0].keys())

        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)

        logger.info(f"Manifest written: {manifest_path} ({len(manifest_rows)} rows)")

    # Step 4: Print summary
    console.print()
    table = Table(title="Feature Extraction Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Total Recordings", str(len(recordings)))
    table.add_row("Processed", str(processed))
    table.add_row("Skipped (already exist)", str(skipped))
    table.add_row("Failed (missing/corrupt)", str(failed))
    table.add_row("Total Segments", str(total_segments))
    table.add_row("Silent Segments", str(skipped_silent))
    table.add_row("Manifest Rows", str(len(manifest_rows)))
    table.add_row("Elapsed Time", f"{elapsed:.1f}s")
    table.add_row(
        "Speed",
        f"{processed / elapsed:.1f} recordings/sec" if elapsed > 0 else "N/A",
    )

    console.print(table)

    # Disk usage
    features_path = Path(config.features_dir)
    mel_size = sum(f.stat().st_size for f in features_path.glob("mel/*.npy"))
    mfcc_size = sum(f.stat().st_size for f in features_path.glob("mfcc/*.npy"))
    total_size = mel_size + mfcc_size

    console.print(
        f"\n  Disk Usage: {total_size / 1024 / 1024:.1f} MB "
        f"(mel: {mel_size / 1024 / 1024:.1f} MB, "
        f"mfcc: {mfcc_size / 1024 / 1024:.1f} MB)"
    )

    return {
        "total": len(recordings),
        "processed": processed,
        "failed": failed,
        "segments": total_segments,
        "silent": skipped_silent,
        "elapsed_seconds": round(elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="PRISM Audio Feature Extraction Pipeline"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["coughvid", "coswara", "icbhi"],
        help="Process only a specific dataset (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of recordings to process (for testing)",
    )

    args = parser.parse_args()
    run_feature_extraction(dataset_name=args.dataset, limit=args.limit)


if __name__ == "__main__":
    main()
