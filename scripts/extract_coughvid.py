"""
extract_coughvid.py – One‑time extraction for the COUGHVID dataset.

The COUGHVID archive (coughvid.zip) contains audio files (.webm, .wav, .ogg)
directly under public_dataset_v3/coughvid_20211012/.  This script extracts
only the audio files (skipping .json metadata) to datasets/raw/coughvid/.

Usage (from the repository root):
    poetry run python scripts/extract_coughvid.py

After running, the audio files will be at:
    datasets/raw/coughvid/<uuid>.webm
    datasets/raw/coughvid/<uuid>.wav
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

console = Console()

ROOT = Path(__file__).resolve().parent.parent
COUGHVID_ZIP = ROOT / "datasets" / "raw" / "coughvid.zip"
EXTRACT_DIR = ROOT / "datasets" / "raw" / "coughvid"
MARKER_FILE = EXTRACT_DIR / ".extraction_complete"

AUDIO_EXTENSIONS = {".webm", ".wav", ".ogg", ".mp3", ".flac"}


def extract_coughvid() -> None:
    """Extract audio files from coughvid.zip to datasets/raw/coughvid/."""

    if MARKER_FILE.exists():
        console.print(
            "[green]COUGHVID already extracted (marker file found). Skipping.[/green]"
        )
        return

    if not COUGHVID_ZIP.exists():
        console.print(f"[red]COUGHVID ZIP not found: {COUGHVID_ZIP}[/red]")
        return

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold cyan]Extracting COUGHVID dataset[/bold cyan]")
    console.print(f"  Source : {COUGHVID_ZIP}")
    console.print(f"  Target : {EXTRACT_DIR}\n")

    with zipfile.ZipFile(COUGHVID_ZIP, "r") as zf:
        # Filter to only audio files
        audio_entries = [
            name
            for name in zf.namelist()
            if Path(name).suffix.lower() in AUDIO_EXTENSIONS and not name.endswith("/")
        ]

        console.print(f"  Audio files in ZIP: {len(audio_entries)}\n")

        with Progress(console=console) as progress:
            task = progress.add_task("Extracting audio...", total=len(audio_entries))

            for entry in audio_entries:
                # Extract to flat structure (strip the nested directory prefix)
                basename = Path(entry).name
                target = EXTRACT_DIR / basename

                if not target.exists():
                    data = zf.read(entry)
                    target.write_bytes(data)

                progress.advance(task)

    # Count extracted files
    count = sum(
        1
        for _ in EXTRACT_DIR.iterdir()
        if _.is_file() and _.suffix.lower() in AUDIO_EXTENSIONS
    )
    console.print("\n[green]Extraction complete![/green]")
    console.print(f"  Audio files extracted: {count}")
    console.print(f"  Location: {EXTRACT_DIR}")

    MARKER_FILE.write_text(f"Extracted {count} audio files\n")


if __name__ == "__main__":
    extract_coughvid()
