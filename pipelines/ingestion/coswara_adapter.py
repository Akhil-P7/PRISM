"""
PRISM Pipelines — Coswara Dataset Adapter

Reads combined_data.csv from the Coswara ZIP archive and maps
columns to the unified PRISM schema.

Raw columns used:
    id, a (age), g (gender), covid_status, smoker, fever, cough,
    asthma, diabetes, ht

Mapping:
    source_subject_id      <- id
    age                    <- a (int)
    gender                 <- g ('male'/'female')
    respiratory_condition  <- covid_status ('healthy', 'positive_mild', etc.)
    has_fever              <- fever (boolean-ish)
    is_smoker              <- smoker (boolean-ish)
"""

import zipfile
from typing import Any

import pandas as pd
from loguru import logger

from pipelines.ingestion.base_adapter import BaseAdapter


class CoswaraAdapter(BaseAdapter):
    """Adapter for the Coswara dataset."""

    METADATA_PATH = "combined_data.csv"

    def __init__(self):
        super().__init__(
            dataset_name="Coswara",
            version="1.0",
            description=(
                "Coswara respiratory sound dataset collected by IISc Bangalore. "
                "Contains cough, breathing, and voice samples for COVID-19 screening."
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

    def _normalize_condition(self, covid_status: Any) -> str | None:
        if pd.isna(covid_status):
            return None
        val = str(covid_status).strip().lower()
        mapping = {
            "healthy": "Healthy",
            "positive_mild": "COVID-19",
            "positive_moderate": "COVID-19",
            "positive_asymp": "COVID-19",
            "resp_illness_not_identified": "Respiratory Illness",
            "recovered_full": "Recovered",
            "no_resp_illness_exposed": "Exposed",
        }
        return mapping.get(val, val.replace("_", " ").title())

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
        return val in ("true", "1", "yes", "t")

    def extract_metadata(self, zip_path: str) -> list[dict[str, Any]]:
        logger.info(f"Reading Coswara metadata from {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as z, z.open(self.METADATA_PATH) as f:
            df = pd.read_csv(f)

        logger.info(f"Loaded {len(df)} rows from combined_data.csv")

        subjects: list[dict[str, Any]] = []

        for _, row in df.iterrows():
            subject_id = str(row.get("id", ""))
            if not subject_id or subject_id == "nan":
                continue

            # Coswara stores audio per-subject in folders named by subject ID.
            # Audio types: cough-shallow, cough-heavy, breathing-shallow,
            # breathing-deep, vowel-a, vowel-e, vowel-o, counting-normal, counting-fast
            audio_types = [
                "cough-shallow",
                "cough-heavy",
            ]

            recordings = []
            for audio_type in audio_types:
                recordings.append(
                    {
                        "file_path": f"datasets/raw/coswara/{subject_id}/{audio_type}",
                        "duration": None,
                        "equipment": "Smartphone",
                        "is_cough": True,
                    }
                )

            subjects.append(
                {
                    "source_subject_id": subject_id,
                    "age": self._safe_int(row.get("a")),
                    "gender": self._normalize_gender(row.get("g")),
                    "respiratory_condition": self._normalize_condition(
                        row.get("covid_status")
                    ),
                    "has_fever": self._safe_bool(row.get("fever")),
                    "is_smoker": self._safe_bool(row.get("smoker")),
                    "recordings": recordings,
                }
            )

        logger.info(
            f"Coswara: {len(subjects)} subjects, "
            f"{sum(len(s['recordings']) for s in subjects)} recordings"
        )
        return subjects
