"""
extract_coswara.py – One‑time extraction for the Coswara dataset.

The Coswara archive (coswara.zip) contains date‑grouped folders, each
holding split tarballs (e.g., 20200413.tar.gz.aa, .ab, …).  The actual
audio files (.wav) are buried inside those tarballs.

This script:
    1. Opens coswara.zip
    2. For each date folder, extracts the split parts to a temp directory
    3. Concatenates the parts into a single .tar.gz
    4. Extracts the tarball to datasets/raw/coswara/
    5. Cleans up temp files

Usage (from the repository root):
    poetry run python scripts/extract_coswara.py

After running, the audio files will be at:
    datasets/raw/coswara/<subject_id>/cough-shallow.wav
    datasets/raw/coswara/<subject_id>/cough-heavy.wav
"""

from __future__ import annotations

import tarfile
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.progress import Progress

console = Console()

ROOT = Path(__file__).resolve().parent.parent
COSWARA_ZIP = ROOT / "datasets" / "raw" / "coswara.zip"
EXTRACT_DIR = ROOT / "datasets" / "raw" / "coswara"
MARKER_FILE = EXTRACT_DIR / ".extraction_complete"


def extract_coswara() -> None:
    """Extract all Coswara audio from the nested ZIP → split‑tarball structure."""

    if MARKER_FILE.exists():
        console.print(
            "[green]Coswara already extracted (marker file found). Skipping.[/green]"
        )
        return

    if not COSWARA_ZIP.exists():
        console.print(f"[red]Coswara ZIP not found: {COSWARA_ZIP}[/red]")
        return

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold cyan]Extracting Coswara dataset[/bold cyan]")
    console.print(f"  Source : {COSWARA_ZIP}")
    console.print(f"  Target : {EXTRACT_DIR}\n")

    with zipfile.ZipFile(COSWARA_ZIP, "r") as zf:
        # Group split parts by their archive base name
        # e.g. "20200413/20200413.tar.gz.aa" → base "20200413/20200413.tar.gz"
        archives: dict[str, list[str]] = defaultdict(list)
        for name in zf.namelist():
            if ".tar.gz." in name and not name.endswith("/"):
                # Strip the last suffix (.aa, .ab, etc.) to get the base
                base = name.rsplit(".", 1)[0]  # "20200413/20200413.tar.gz"
                archives[base].append(name)

        if not archives:
            console.print("[yellow]No split tarballs found in coswara.zip[/yellow]")
            return

        console.print(f"  Found {len(archives)} date archives\n")

        # Sort the parts for each archive
        for base in archives:
            archives[base] = sorted(archives[base])

        with Progress(console=console) as progress:
            task = progress.add_task("Extracting archives...", total=len(archives))

            for base, parts in archives.items():
                date_label = base.split("/")[0]
                progress.update(task, description=f"Processing {date_label}...")

                # Create a temp dir for concatenation
                with tempfile.TemporaryDirectory(dir=str(ROOT)) as tmpdir:
                    tmp_tar_path = Path(tmpdir) / "combined.tar.gz"

                    # Step 1: Extract split parts from ZIP and concatenate
                    logger.info(f"Concatenating {len(parts)} parts for {date_label}")
                    with open(tmp_tar_path, "wb") as out_f:
                        for part_name in parts:
                            part_bytes = zf.read(part_name)
                            out_f.write(part_bytes)

                    # Step 2: Extract the combined tarball
                    logger.info(
                        f"Extracting tarball for {date_label} ({tmp_tar_path.stat().st_size / 1024 / 1024:.1f} MB)"
                    )
                    try:
                        with tarfile.open(tmp_tar_path, "r:gz") as tar:
                            # Extract to the coswara directory
                            tar.extractall(path=EXTRACT_DIR, filter="data")
                    except tarfile.ReadError as e:
                        logger.warning(f"Failed to extract {date_label}: {e}")
                        progress.advance(task)
                        continue

                progress.advance(task)

    # Count extracted audio files
    wav_count = sum(1 for _ in EXTRACT_DIR.rglob("*.wav"))
    console.print("\n[green]Extraction complete![/green]")
    console.print(f"  Audio files found: {wav_count}")
    console.print(f"  Location: {EXTRACT_DIR}")

    # Write marker file
    MARKER_FILE.write_text(f"Extracted {wav_count} audio files\n")


if __name__ == "__main__":
    extract_coswara()
