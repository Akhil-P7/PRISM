"""
PRISM Pipelines — COUGHVID Dataset Adapter

Reads metadata_compiled.csv from the COUGHVID ZIP archive and maps
columns to the unified PRISM schema.

Raw columns used:
    uuid, age, gender, respiratory_condition, fever_muscle_pain, status,
    cough_detected

Mapping:
    source_subject_id  <- uuid
    age                <- age (float -> int)
    gender             <- gender ('male'/'female')
    respiratory_condition <- status ('healthy', 'symptomatic', 'COVID-19')
    has_fever          <- fever_muscle_pain (bool)
    is_smoker          <- None (not available in COUGHVID)
    file_path          <- uuid + '.webm' (audio stored as WebM)
    is_cough           <- cough_detected > 0.5
"""

import zipfile
from typing import Any

import pandas as pd
from loguru import logger

from pipelines.ingestion.base_adapter import BaseAdapter


class CoughvidAdapter(BaseAdapter):
    """Adapter for the COUGHVID v3 dataset."""

    METADATA_PATH = "public_dataset_v3/coughvid_20211012/metadata_compiled.csv"

    def __init__(self):
        super().__init__(
            dataset_name="COUGHVID",
            version="3.0",
            description=(
                "COUGHVID crowdsourced cough recording dataset with "
                "expert annotations. ~34K recordings via web app."
            ),
        )

    def _normalize_gender(self, value: Any) -> str | None:
        if pd.isna(value):
            return None
        val = str(value).strip().lower()
        if val in ("male", "m"):
            return "Male"
        if val in ("female", "f"):
            return "Female"
        return val.capitalize() if val else None

    def _normalize_condition(self, status: Any) -> str | None:
        if pd.isna(status):
            return None
        val = str(status).strip().lower()
        mapping = {
            "healthy": "Healthy",
            "symptomatic": "Symptomatic",
            "covid-19": "COVID-19",
        }
        return mapping.get(val, val.capitalize())

    def _safe_int(self, value: Any) -> int | None:
        if pd.isna(value):
            return None
        try:
            v = int(float(value))
            return v if 0 < v < 120 else None
        except (ValueError, TypeError):
            return None

    def _safe_bool(self, value: Any) -> bool | None:
        if pd.isna(value):
            return None
        if isinstance(value, bool):
            return value
        val = str(value).strip().lower()
        return val in ("true", "1", "yes")

    def extract_metadata(self, zip_path: str) -> list[dict[str, Any]]:
        logger.info(f"Reading COUGHVID metadata from {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as z, z.open(self.METADATA_PATH) as f:
            df = pd.read_csv(f)

        logger.info(f"Loaded {len(df)} rows from metadata_compiled.csv")

        subjects: dict[str, dict] = {}

        for _, row in df.iterrows():
            uid = str(row.get("uuid", ""))
            if not uid or uid == "nan":
                continue

            if uid not in subjects:
                subjects[uid] = {
                    "source_subject_id": uid,
                    "age": self._safe_int(row.get("age")),
                    "gender": self._normalize_gender(row.get("gender")),
                    "respiratory_condition": self._normalize_condition(
                        row.get("status")
                    ),
                    "has_fever": self._safe_bool(row.get("fever_muscle_pain")),
                    "is_smoker": None,  # Not available in COUGHVID
                    "recordings": [],
                }

            # Each row in COUGHVID corresponds to one recording
            cough_detected = row.get("cough_detected", 0)
            is_cough = bool(not pd.isna(cough_detected) and float(cough_detected) > 0.5)

            subjects[uid]["recordings"].append(
                {
                    "file_path": f"datasets/raw/coughvid/{uid}.webm",
                    "duration": None,  # Not directly in metadata
                    "equipment": "Smartphone",
                    "is_cough": is_cough,
                }
            )

        result = list(subjects.values())
        logger.info(
            f"COUGHVID: {len(result)} unique subjects, "
            f"{sum(len(s['recordings']) for s in result)} recordings"
        )
        return result
