"""
PRISM Pipelines — Audio Extractor

Reads audio files directly from ZIP archives and returns numpy waveform
arrays resampled to the target sample rate.  No files are extracted to
disk — everything is processed in memory.

Supports the three dataset layouts:
    COUGHVID : public_dataset_v3/coughvid_20211012/{uuid}.webm
    Coswara  : {subject_id}/{audio_type}.wav  (also .ogg)
    ICBHI    : ICBHI_final_database/{filename}.wav

Usage:
    extractor = AudioExtractor(config)
    waveform = extractor.load("datasets/raw/icbhi/101_1b1_Al_sc_Meditron.wav",
                              dataset_name="ICBHI")
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import librosa
import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from pipelines.preprocessing.audio_config import AudioConfig


class AudioExtractor:
    """Loads audio from ZIP archives into numpy arrays."""

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self._zip_handles: dict[str, zipfile.ZipFile] = {}
        self._extracted_paths: dict[str, Path] = {}
        # Build lookup of files inside each ZIP (cached for performance)
        self._zip_listings: dict[str, set[str]] = {}
        # Pre‑compute extracted folder locations (if they exist)
        for name, zip_path in self.config.zip_paths.items():
            extracted_dir = Path(zip_path).with_suffix("")  # remove .zip
            if extracted_dir.is_dir():
                self._extracted_paths[name] = extracted_dir

    def _get_zip(self, zip_path: str) -> zipfile.ZipFile:
        """Get or open a ZIP file handle (kept open for batch processing)."""
        if zip_path not in self._zip_handles:
            self._zip_handles[zip_path] = zipfile.ZipFile(zip_path, "r")
            self._zip_listings[zip_path] = set(self._zip_handles[zip_path].namelist())
        return self._zip_handles[zip_path]

    def _find_in_zip(self, zip_path: str, filename: str) -> str | None:
        """Find a file inside a ZIP, trying common path prefixes."""
        self._get_zip(zip_path)
        listing = self._zip_listings[zip_path]

        # Direct match
        if filename in listing:
            return filename

        # Try common subdirectory prefixes per dataset
        prefixes = [
            "ICBHI_final_database/",
            "public_dataset_v3/coughvid_20211012/",
            "",
        ]
        for prefix in prefixes:
            candidate = prefix + filename
            if candidate in listing:
                return candidate

        # Fuzzy: search for the basename anywhere in the ZIP
        basename = filename.split("/")[-1]
        for entry in listing:
            if entry.endswith("/" + basename) or entry == basename:
                return entry

        return None

    def _resolve_zip_and_member(
        self, file_path: str, dataset_name: str
    ) -> tuple[Path, str]:
        """Resolve the storage location for a recording.

        Returns a tuple ``(base_path, member_path)`` where ``base_path`` is either:
        * a Path to the ZIP file (if the dataset is still archived), or
        * a Path to the extracted folder (if we have run the one‑time extraction).
        ``member_path`` is the relative path inside that archive/folder.
        """
        dataset_key = dataset_name.lower().replace("-", "")
        # Determine if we have an extracted folder for this dataset
        if dataset_key in self._extracted_paths:
            base_path = self._extracted_paths[dataset_key]
        else:
            raw_path = self.config.zip_paths.get(dataset_key)
            if not raw_path:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            base_path = Path(raw_path)

        # Build the relative member path from the DB file_path
        parts = file_path.replace("\\", "/").split("/")
        # Default: just the filename (ICBHI)
        member_path = parts[-1]
        if dataset_key == "coswara" and len(parts) >= 3:
            # Expected layout: datasets/raw/coswara/<subject>/<audio_name>
            subject_id = parts[-2]
            audio_name = parts[-1]
            member_path = f"{subject_id}/{audio_name}"
        return base_path, member_path

    def _find_coswara_audio(self, zip_path: str, member_hint: str) -> str | None:
        """
        Coswara audio files can have varying extensions (.wav, .ogg, .mp3).
        Try common extensions to find the actual file.
        """
        self._get_zip(zip_path)
        listing = self._zip_listings[zip_path]

        # member_hint is like "subject_id/cough-shallow"
        for ext in (".wav", ".ogg", ".mp3", ".webm", ".flac"):
            candidate = member_hint + ext
            if candidate in listing:
                return candidate

        # Also try without extension (some entries may already include it)
        if member_hint in listing:
            return member_hint

        return None

    _AUDIO_EXTENSIONS = (".wav", ".ogg", ".webm", ".mp3", ".flac")

    def _build_subject_index(self, base_dir: Path) -> dict[str, Path]:
        """
        Build a mapping from subject_id → full directory path.

        Coswara extracts to: coswara/20200413/subject_id/cough-heavy.wav
        But the DB stores:   coswara/subject_id/cough-heavy
        This index resolves the date-folder indirection.
        """
        index: dict[str, Path] = {}
        for date_dir in base_dir.iterdir():
            if date_dir.is_dir() and date_dir.name.startswith("20"):
                for subj_dir in date_dir.iterdir():
                    if subj_dir.is_dir():
                        index[subj_dir.name] = subj_dir
        if index:
            logger.info(
                f"Built subject index: {len(index)} subjects in {base_dir.name}"
            )
        return index

    def _find_on_disk(self, base_dir: Path, member_path: str) -> Path | None:
        """
        Locate an audio file on disk, trying multiple extensions and
        subdirectory structures.

        Handles three cases:
        - Direct path match (ICBHI)
        - Coswara: DB stores 'subj_id/cough-shallow' but files are at
          'date/subj_id/cough-shallow.wav'
        - COUGHVID: DB stores one extension but file may exist with another
        """
        candidate = base_dir / member_path

        # Exact match
        if candidate.is_file():
            return candidate

        # Try appending extensions (Coswara case — no extension in DB)
        for ext in self._AUDIO_EXTENSIONS:
            test = (
                candidate.with_suffix(ext)
                if candidate.suffix
                else Path(str(candidate) + ext)
            )
            if test.is_file():
                return test

        # Try swapping extension (COUGHVID case — DB says .webm but file is .wav)
        if candidate.suffix:
            stem = candidate.with_suffix("")
            for ext in self._AUDIO_EXTENSIONS:
                test = stem.with_suffix(ext)
                if test.is_file():
                    return test

        # Subdirectory search (Coswara date-folder structure)
        # member_path is like "subject_id/cough-heavy"
        parts = member_path.replace("\\", "/").split("/")
        if len(parts) >= 2:
            subject_id = parts[0]
            audio_name = "/".join(parts[1:])

            # Build index on first use
            cache_key = str(base_dir)
            if not hasattr(self, "_subject_index"):
                self._subject_index: dict[str, dict[str, Path]] = {}
            if cache_key not in self._subject_index:
                self._subject_index[cache_key] = self._build_subject_index(base_dir)

            idx = self._subject_index.get(cache_key, {})
            subj_dir = idx.get(subject_id)
            if subj_dir:
                for ext in self._AUDIO_EXTENSIONS:
                    test = subj_dir / (audio_name + ext)
                    if test.is_file():
                        return test
                # Try exact name (already has extension)
                test = subj_dir / audio_name
                if test.is_file():
                    return test

        return None

    def load(
        self,
        file_path: str,
        dataset_name: str,
    ) -> np.ndarray | None:
        """
        Load audio from a ZIP archive and return a 1-D numpy array
        resampled to config.sample_rate.

        Returns None if the audio cannot be loaded (missing file, corrupt, etc.).
        """
        try:
            base_path, member_path = self._resolve_zip_and_member(
                file_path, dataset_name
            )

            # Load from extracted folder if available
            if isinstance(base_path, Path) and base_path.is_dir():
                # Direct filesystem access — try the exact path first,
                # then try common audio extensions (Coswara paths have no
                # extension, COUGHVID may have mismatched extensions in DB)
                audio_file = self._find_on_disk(base_path, member_path)
                if audio_file is None:
                    logger.debug(f"Audio not found on disk: {base_path / member_path}")
                    return None
                waveform, sr = librosa.load(
                    str(audio_file), sr=self.config.sample_rate, mono=True
                )
            else:
                # ZIP based access
                zip_path = str(base_path)
                # Find the actual file inside the ZIP
                if dataset_name.lower() == "coswara":
                    member = self._find_coswara_audio(zip_path, member_path)
                else:
                    member = self._find_in_zip(zip_path, member_path)

                if member is None:
                    logger.debug(f"Audio not found in ZIP: {file_path}")
                    return None

                z = self._get_zip(zip_path)
                audio_bytes = z.read(member)

                # Load with librosa from in-memory bytes
                buf = io.BytesIO(audio_bytes)
                waveform, sr = librosa.load(
                    buf,
                    sr=self.config.sample_rate,
                    mono=True,
                )

            # Validate: reject silence or very short clips
            if len(waveform) < self.config.sample_rate * 0.1:  # < 100ms
                logger.debug(f"Audio too short ({len(waveform)} samples): {file_path}")
                return None

            return waveform

        except Exception as e:
            logger.warning(f"Failed to load audio {file_path}: {e}")
            return None

    def close(self) -> None:
        """Close all open ZIP handles."""
        for z in self._zip_handles.values():
            z.close()
        self._zip_handles.clear()
        self._zip_listings.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
