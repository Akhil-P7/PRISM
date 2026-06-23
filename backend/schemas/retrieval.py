"""
PRISM Backend — Retrieval API Schemas

Pydantic models for the similarity search endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class SimilaritySearchRequest(BaseModel):
    """Request body for similarity search."""

    embedding: list[float] = Field(
        ...,
        min_length=512,
        max_length=512,
        description="512-dimensional L2-normalised embedding vector.",
    )
    k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return.",
    )
    group_by: str | None = Field(
        default=None,
        description=(
            "Optional grouping mode. Set to 'subject' to aggregate results "
            "by patient (subject_id) instead of returning raw segments."
        ),
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class SegmentResult(BaseModel):
    """A single segment-level similarity result."""

    embedding_idx: int = Field(description="Index in the embedding matrix.")
    segment_id: int | str = Field(description="Segment identifier.")
    subject_id: str = Field(description="Patient subject ID.")
    recording_id: str = Field(description="Source recording ID.")
    is_cough: bool = Field(description="Whether this segment is a cough.")
    similarity_score: float = Field(
        description="Cosine similarity score (0–1, higher = more similar)."
    )


class PatientResult(BaseModel):
    """An aggregated patient-level similarity result."""

    subject_id: str = Field(description="Patient subject ID.")
    best_similarity: float = Field(
        description="Highest similarity score among matched segments."
    )
    avg_similarity: float = Field(
        description="Average similarity across matched segments."
    )
    num_matching_segments: int = Field(
        description="Number of segments matched for this patient."
    )
    cough_ratio: float = Field(
        description="Fraction of matched segments that are coughs (0–1)."
    )
    top_segments: list[SegmentResult] = Field(
        default_factory=list,
        description="Top 3 most similar segments for this patient.",
    )


class SimilaritySearchResponse(BaseModel):
    """Response wrapper for similarity search results."""

    query_mode: str = Field(description="Search mode: 'segment' or 'subject'.")
    total_results: int = Field(description="Number of results returned.")
    results: list[SegmentResult] | list[PatientResult] = Field(
        description="Search results (segments or patients depending on query_mode)."
    )


class IndexHealthResponse(BaseModel):
    """Response for the index health check."""

    index_loaded: bool = Field(description="Whether the index is loaded.")
    num_vectors: int = Field(
        default=0,
        description="Number of vectors in the index.",
    )
    message: str = Field(
        default="",
        description="Status message.",
    )
