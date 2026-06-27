"""
PRISM Pipelines — Base Dataset Adapter

Abstract base class that defines the contract for all dataset adapters.
Each adapter must implement `extract_metadata()` which reads the raw
dataset source and returns unified Subject + Recording rows.
"""

import uuid
from abc import ABC, abstractmethod
from typing import Any

from loguru import logger
from sqlalchemy.orm import Session

from database.models.dataset import Dataset
from database.models.recording import Recording
from database.models.subject import Subject


class BaseAdapter(ABC):
    """Abstract base for dataset ingestion adapters."""

    def __init__(
        self, dataset_name: str = "", version: str = "", description: str = ""
    ):
        self.dataset_name = dataset_name
        self.version = version
        self.description = description

    @abstractmethod
    def extract_metadata(self, zip_path: str) -> list[dict[str, Any]]:
        """
        Extract metadata from the raw dataset source.

        Must return a list of dicts, each with keys:
            - source_subject_id: str
            - age: int | None
            - gender: str | None
            - respiratory_condition: str | None
            - has_fever: bool | None
            - is_smoker: bool | None
            - recordings: list[dict] where each dict has:
                - file_path: str
                - duration: float | None
                - equipment: str | None
                - is_cough: bool | None
        """
        ...

    def ingest(self, zip_path: str, session: Session) -> dict[str, Any]:
        """
        Run the full ingestion pipeline:
        1. Register the dataset
        2. Extract metadata from ZIP
        3. Insert subjects and recordings into the database

        Returns a summary dict with counts.
        """
        logger.info(f"Starting ingestion for {self.dataset_name}...")

        # Step 1: Register dataset (or get existing)
        dataset = session.query(Dataset).filter_by(name=self.dataset_name).first()
        if dataset is None:
            dataset = Dataset(
                id=uuid.uuid4(),
                name=self.dataset_name,
                version=self.version,
                description=self.description,
            )
            session.add(dataset)
            session.flush()
            logger.info(f"Registered new dataset: {self.dataset_name}")
        else:
            logger.info(
                f"Dataset '{self.dataset_name}' already exists, skipping registration."
            )

        # Step 2: Extract metadata
        logger.info(f"Extracting metadata from {zip_path}...")
        subjects_data = self.extract_metadata(zip_path)
        logger.info(f"Extracted {len(subjects_data)} subjects")

        # Step 3: Insert into database
        subject_count = 0
        recording_count = 0

        for subj_data in subjects_data:
            # Check if subject already exists
            existing = (
                session.query(Subject)
                .filter_by(
                    dataset_id=dataset.id,
                    source_subject_id=subj_data["source_subject_id"],
                )
                .first()
            )
            if existing:
                existing.age = subj_data.get("age")
                existing.gender = subj_data.get("gender")
                existing.respiratory_condition = subj_data.get("respiratory_condition")
                existing.has_fever = subj_data.get("has_fever")
                existing.is_smoker = subj_data.get("is_smoker")
                session.flush()
                subject_count += 1
                continue

            subject = Subject(
                id=uuid.uuid4(),
                dataset_id=dataset.id,
                source_subject_id=subj_data["source_subject_id"],
                age=subj_data.get("age"),
                gender=subj_data.get("gender"),
                respiratory_condition=subj_data.get("respiratory_condition"),
                has_fever=subj_data.get("has_fever"),
                is_smoker=subj_data.get("is_smoker"),
            )
            session.add(subject)
            session.flush()
            subject_count += 1

            for rec_data in subj_data.get("recordings", []):
                recording = Recording(
                    id=uuid.uuid4(),
                    subject_id=subject.id,
                    file_path=rec_data["file_path"],
                    duration=rec_data.get("duration"),
                    equipment=rec_data.get("equipment"),
                    is_cough=rec_data.get("is_cough"),
                )
                session.add(recording)
                recording_count += 1

        session.commit()

        summary = {
            "dataset": self.dataset_name,
            "subjects_inserted": subject_count,
            "recordings_inserted": recording_count,
        }
        logger.info(
            f"Ingestion complete: {subject_count} subjects, "
            f"{recording_count} recordings"
        )
        return summary
