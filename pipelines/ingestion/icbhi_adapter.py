"""
PRISM Pipelines — ICBHI Dataset Adapter

Parses the ICBHI Respiratory Sound Database from its ZIP archive.
ICBHI has NO CSV metadata file — all metadata is encoded in filenames.

Filename format (from filename_format.txt):
    {PatientNumber}_{RecordingIndex}{BodyLocation}_{AcquisitionMode}_{Equipment}

    Example: 101_1b1_Al_sc_Meditron.wav
        PatientNumber = 101
        RecordingIndex = 1b1
        BodyLocation = Al (Anterior left)
        AcquisitionMode = sc (single channel)
        Equipment = Meditron

Annotation TXT files (same name as WAV):
    Each line: start_time  end_time  crackle(0/1)  wheeze(0/1)
    These are respiratory cycle boundaries with labels.

Body Location Codes:
    Tc  = Trachea
    Al  = Anterior left
    Ar  = Anterior right
    Pl  = Posterior left
    Pr  = Posterior right
    Ll  = Lateral left
    Lr  = Lateral right

Equipment Codes:
    Meditron   = WelchAllyn Meditron Master Elite Electronic Stethoscope
    LittC2SE   = 3M Littmann Classic II SE Stethoscope
    Litt3200   = 3M Littmann 3200 Electronic Stethoscope
    AKGC417L   = AKG C417L Microphone
"""

import os
import re
import zipfile
from collections import defaultdict
from typing import Any

from loguru import logger

from pipelines.ingestion.base_adapter import BaseAdapter

# Map equipment codes to human-readable names
EQUIPMENT_MAP = {
    "Meditron": "WelchAllyn Meditron",
    "LittC2SE": "3M Littmann Classic II SE",
    "Litt3200": "3M Littmann 3200",
    "AKGC417L": "AKG C417L Microphone",
}

# Map body location codes
LOCATION_MAP = {
    "Tc": "Trachea",
    "Al": "Anterior Left",
    "Ar": "Anterior Right",
    "Pl": "Posterior Left",
    "Pr": "Posterior Right",
    "Ll": "Lateral Left",
    "Lr": "Lateral Right",
}


class IcbhiAdapter(BaseAdapter):
    """Adapter for the ICBHI Respiratory Sound Database."""

    # Regex to parse ICBHI filenames
    FILENAME_PATTERN = re.compile(
        r"^(\d+)_(.+?)_([A-Z][a-z])_([a-z]{2})_(\w+)\.(wav|txt)$"
    )

    def __init__(self):
        super().__init__(
            dataset_name="ICBHI",
            version="2017",
            description=(
                "ICBHI 2017 Respiratory Sound Database. "
                "920 recordings from 126 patients with stethoscope-captured "
                "lung sounds annotated for crackles and wheezes."
            ),
        )

    def _parse_filename(self, filename: str) -> dict[str, str] | None:
        """Parse an ICBHI filename into its components."""
        basename = os.path.basename(filename)
        match = self.FILENAME_PATTERN.match(basename)
        if not match:
            return None
        return {
            "patient_id": match.group(1),
            "recording_index": match.group(2),
            "body_location": match.group(3),
            "acquisition_mode": match.group(4),
            "equipment_code": match.group(5),
            "extension": match.group(6),
        }

    def _parse_annotations(self, content: str) -> dict[str, Any]:
        """
        Parse an ICBHI annotation file.
        Each line: start_time  end_time  crackle(0/1)  wheeze(0/1)
        Returns summary stats for the recording.
        """
        lines = content.strip().split("\n")
        total_cycles = 0
        crackle_cycles = 0
        wheeze_cycles = 0
        duration = 0.0

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    _ = float(parts[0])
                    end = float(parts[1])
                    crackle = int(parts[2])
                    wheeze = int(parts[3])

                    total_cycles += 1
                    if crackle == 1:
                        crackle_cycles += 1
                    if wheeze == 1:
                        wheeze_cycles += 1
                    duration = max(duration, end)
                except (ValueError, IndexError):
                    continue

        return {
            "duration": round(duration, 3) if duration > 0 else None,
            "total_cycles": total_cycles,
            "crackle_cycles": crackle_cycles,
            "wheeze_cycles": wheeze_cycles,
        }

    def _load_diagnoses(self, zip_path: str) -> dict[str, str]:
        """Load patient diagnoses from the external CSV file."""
        import csv

        # The CSV is located in the extracted folder datasets/raw/icbhi/Patient_diagnosis.csv
        zip_dir = os.path.dirname(zip_path)
        csv_path = os.path.join(zip_dir, "icbhi", "Patient_diagnosis.csv")

        diagnoses = {}
        if os.path.exists(csv_path):
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        diagnoses[str(row[0]).strip()] = row[1].strip()
            logger.info(f"Loaded {len(diagnoses)} diagnoses from {csv_path}")
        else:
            logger.warning(
                f"Diagnosis file not found at {csv_path}. Using None for condition."
            )

        return diagnoses

    def extract_metadata(self, zip_path: str) -> list[dict[str, Any]]:
        logger.info(f"Reading ICBHI metadata from {zip_path}")
        diagnoses = self._load_diagnoses(zip_path)

        # Group recordings by patient
        patients: dict[str, list[dict]] = defaultdict(list)

        with zipfile.ZipFile(zip_path, "r") as z:
            wav_files = [f for f in z.namelist() if f.endswith(".wav")]
            logger.info(f"Found {len(wav_files)} WAV files")

            for wav_path in wav_files:
                parsed = self._parse_filename(wav_path)
                if not parsed:
                    continue

                patient_id = parsed["patient_id"]
                equipment = EQUIPMENT_MAP.get(
                    parsed["equipment_code"], parsed["equipment_code"]
                )
                location = LOCATION_MAP.get(
                    parsed["body_location"], parsed["body_location"]
                )

                # Try to read the matching annotation file
                txt_path = wav_path.replace(".wav", ".txt")
                annotation = {"duration": None}
                try:
                    content = z.read(txt_path).decode("utf-8", errors="replace")
                    annotation = self._parse_annotations(content)
                except KeyError:
                    pass

                patients[patient_id].append(
                    {
                        "file_path": f"datasets/raw/icbhi/{os.path.basename(wav_path)}",
                        "duration": annotation.get("duration"),
                        "equipment": equipment,
                        "is_cough": False,  # ICBHI is lung sounds, not cough
                        "body_location": location,
                        "acquisition_mode": parsed["acquisition_mode"],
                        "crackle_cycles": annotation.get("crackle_cycles", 0),
                        "wheeze_cycles": annotation.get("wheeze_cycles", 0),
                        "total_cycles": annotation.get("total_cycles", 0),
                    }
                )

        # Build unified subject entries
        subjects: list[dict[str, Any]] = []
        for patient_id, recordings in patients.items():
            # Strip extra annotation fields before inserting into DB
            db_recordings = []
            for rec in recordings:
                db_recordings.append(
                    {
                        "file_path": rec["file_path"],
                        "duration": rec["duration"],
                        "equipment": rec["equipment"],
                        "is_cough": rec["is_cough"],
                    }
                )

            subjects.append(
                {
                    "source_subject_id": patient_id,
                    "age": None,  # ICBHI does not include age in the ZIP
                    "gender": None,  # ICBHI does not include gender in the ZIP
                    "respiratory_condition": diagnoses.get(patient_id),
                    "has_fever": None,
                    "is_smoker": None,
                    "recordings": db_recordings,
                }
            )

        logger.info(
            f"ICBHI: {len(subjects)} patients, "
            f"{sum(len(s['recordings']) for s in subjects)} recordings"
        )
        return subjects
