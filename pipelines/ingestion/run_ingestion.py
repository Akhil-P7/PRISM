"""
PRISM Pipelines — Dataset Ingestion Runner

CLI entry point to run all dataset adapters and populate the
PostgreSQL database with metadata from COUGHVID, Coswara, and ICBHI.

Usage:
    poetry run python -m pipelines.ingestion.run_ingestion
    poetry run python -m pipelines.ingestion.run_ingestion --dataset coughvid
    poetry run python -m pipelines.ingestion.run_ingestion --dataset coswara
    poetry run python -m pipelines.ingestion.run_ingestion --dataset icbhi
"""

import argparse
import os
import time
from typing import Any

from loguru import logger
from rich.console import Console
from rich.table import Table

from database.connection import SessionLocal
from pipelines.ingestion.base_adapter import BaseAdapter
from pipelines.ingestion.coswara_adapter import CoswaraAdapter
from pipelines.ingestion.coughvid_adapter import CoughvidAdapter
from pipelines.ingestion.icbhi_adapter import IcbhiAdapter

console = Console()

# Default paths to dataset ZIP files
DATASET_PATHS = {
    "coughvid": os.path.join("datasets", "raw", "coughvid.zip"),
    "coswara": os.path.join("datasets", "raw", "coswara.zip"),
    "icbhi": os.path.join("datasets", "raw", "icbhi.zip"),
}

ADAPTERS: dict[str, type[BaseAdapter]] = {
    "coughvid": CoughvidAdapter,
    "coswara": CoswaraAdapter,
    "icbhi": IcbhiAdapter,
}


def run_ingestion(datasets: list[str] | None = None) -> list[dict[str, Any]]:
    """
    Run ingestion for the specified datasets (or all if None).
    Returns a list of summary dicts.
    """
    if datasets is None:
        datasets = list(ADAPTERS.keys())

    results = []
    session = SessionLocal()

    try:
        for name in datasets:
            if name not in ADAPTERS:
                logger.error(f"Unknown dataset: {name}")
                continue

            zip_path = DATASET_PATHS[name]
            if not os.path.exists(zip_path):
                logger.warning(f"ZIP file not found: {zip_path} — skipping {name}")
                continue

            adapter = ADAPTERS[name]()  # type: ignore[arg-type]
            start_time = time.time()
            summary = adapter.ingest(zip_path, session)
            # Ensure elapsed_seconds can be a float
            summary["elapsed_seconds"] = round(time.time() - start_time, 2)
            results.append(summary)

    except Exception as e:
        session.rollback()
        logger.error(f"Ingestion failed: {e}")
        raise
    finally:
        session.close()

    return results


def display_results(results: list[dict]) -> None:
    """Display ingestion results in a formatted table."""
    table = Table(title="PRISM Dataset Ingestion Results", show_lines=True)
    table.add_column("Dataset", style="cyan", justify="center")
    table.add_column("Subjects", style="green", justify="right")
    table.add_column("Recordings", style="green", justify="right")
    table.add_column("Time (s)", style="yellow", justify="right")

    total_subjects = 0
    total_recordings = 0

    for r in results:
        table.add_row(
            r["dataset"],
            str(r["subjects_inserted"]),
            str(r["recordings_inserted"]),
            str(r["elapsed_seconds"]),
        )
        total_subjects += r["subjects_inserted"]
        total_recordings += r["recordings_inserted"]

    table.add_section()
    table.add_row(
        "TOTAL",
        str(total_subjects),
        str(total_recordings),
        "",
        style="bold",
    )

    console.print()
    console.print(table)
    console.print()


def main():
    parser = argparse.ArgumentParser(description="PRISM Dataset Ingestion Pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(ADAPTERS.keys()),
        default=None,
        help="Ingest a specific dataset. If omitted, all datasets are ingested.",
    )
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else None

    console.print("[bold cyan]PRISM Dataset Ingestion Pipeline[/bold cyan]")
    console.print("=" * 50)

    results = run_ingestion(datasets)
    display_results(results)

    console.print("[bold green]Ingestion complete![/bold green]")


if __name__ == "__main__":
    main()
