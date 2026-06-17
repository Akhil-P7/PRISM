"""
extract_split_archives.py – One‑time extraction utility for Coswara and COUGHVID.

Both datasets are delivered as split tarballs (e.g. 20200413.tar.gz.aa, .ab, …).
This script stitches the parts together, extracts the tarball, and places the
raw audio files under `datasets/raw/<dataset_name>/` preserving the original
folder hierarchy used by the database (`<subject_id>/cough‑shallow.wav`, etc.).

Usage (from the repository root):
    poetry run python scripts/extract_split_archives.py

The script is safe to re‑run – it will skip already‑extracted archives.
"""

import logging
import tarfile
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# Root where the original zip files live (relative to repo root)
ROOT = Path(__file__).parent.parent / "datasets" / "raw"


def find_split_parts(root: Path) -> dict[Path, list[Path]]:
    """Return a mapping from *archive base* (e.g. 20200413.tar.gz) to its split parts.
    The split parts are files ending with ".tar.gz.<suffix>".
    """
    parts: dict[Path, list[Path]] = {}
    for part in root.rglob("*.tar.gz.*"):
        # Example part: 20200413/20200413.tar.gz.aa
        # Strip the last suffix after the final '.' to get the base archive
        base_name = ".".join(part.name.split(".")[:3])  # keep .tar.gz
        archive_path = part.parent / base_name
        parts.setdefault(archive_path, []).append(part)
    # Ensure deterministic order (aa, ab, ...)
    for archive in parts:
        parts[archive] = sorted(parts[archive], key=lambda p: p.name)
    return parts


def concat_parts(archive: Path, part_files: list[Path]) -> Path:
    """Concatenate the split parts into a temporary .tar.gz file and return its path."""
    temp_tar = archive.with_suffix(".tmp.tar.gz")
    logging.info(f"Rebuilding {archive.name} from {len(part_files)} parts")
    with open(temp_tar, "wb") as out_f:
        for part in part_files:
            with open(part, "rb") as in_f:
                out_f.write(in_f.read())
    return temp_tar


def extract_archive(temp_tar: Path, destination: Path) -> None:
    """Extract the temporary tar.gz into `destination` using tqdm for progress."""
    destination.mkdir(parents=True, exist_ok=True)
    logging.info(f"Extracting {temp_tar.name} → {destination}")
    with tarfile.open(temp_tar, "r:gz") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc=f"Extract {temp_tar.stem}"):
            tar.extract(member, path=destination)
    temp_tar.unlink()


def process_dataset(dataset_name: str) -> None:
    """Process a specific dataset (coswara or coughvid).
    The function looks for split tarballs under `ROOT` and extracts them.
    """
    logging.info(f"Processing {dataset_name.upper()} dataset")
    split_parts = find_split_parts(ROOT)
    for archive_path, part_files in split_parts.items():
        # Determine which dataset the archive belongs to by its parent folder name
        # Coswara archives are under a folder named like 20200413, 20200415, ...
        # COUGHVID archives are flat inside `public_dataset_v3` – we treat them as the same logic.
        # We'll simply extract everything; the destination folder will be `<dataset_name>`.
        dest_dir = ROOT / dataset_name
        # Skip if already extracted (check for a marker file)
        marker = dest_dir / f".{archive_path.name}.extracted"
        if marker.exists():
            logging.info(f"Skipping already extracted {archive_path.name}")
            continue
        temp_tar = concat_parts(archive_path, part_files)
        extract_archive(temp_tar, dest_dir)
        # Place a simple marker file to avoid re‑extracting next run
        marker.touch()
        logging.info(f"Finished extracting {archive_path.name}")


def main() -> None:
    # Coswara and COUGHVID use the same split‑tar layout – we just run both.
    for ds in ["coswara", "coughvid"]:
        process_dataset(ds)
    logging.info("All extractions complete.")


if __name__ == "__main__":
    main()
