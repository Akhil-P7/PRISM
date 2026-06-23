"""
PRISM Backend — Retrieval API Router

Endpoints for querying the TurboVec similarity index.

Routes:
    POST /retrieval/search   — Find similar segments or patients
    GET  /retrieval/health    — Check index status
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from loguru import logger

from backend.schemas.retrieval import (
    IndexHealthResponse,
    PatientResult,
    SegmentResult,
    SimilaritySearchRequest,
    SimilaritySearchResponse,
)
from backend.services import retrieval_service

router = APIRouter(prefix="/retrieval", tags=["Retrieval"])


# ---------------------------------------------------------------------------
# POST /retrieval/search
# ---------------------------------------------------------------------------


@router.post(
    "/search",
    response_model=SimilaritySearchResponse,
    summary="Similarity Search",
    description=(
        "Find segments or patients most similar to a query embedding. "
        "Set `group_by='subject'` to aggregate results by patient."
    ),
)
async def search_similar(request: SimilaritySearchRequest) -> SimilaritySearchResponse:
    """Execute a similarity search against the embedding index."""
    try:
        if request.group_by == "subject":
            # Patient-level aggregation
            raw_patients = retrieval_service.find_similar_patients(
                embedding=request.embedding,
                k=request.k,
            )
            patient_results = [
                PatientResult(
                    subject_id=p.subject_id,
                    best_similarity=p.best_similarity,
                    avg_similarity=p.avg_similarity,
                    num_matching_segments=p.num_matching_segments,
                    cough_ratio=p.cough_ratio,
                    top_segments=[
                        SegmentResult(
                            embedding_idx=s.embedding_idx,
                            segment_id=s.segment_id,
                            subject_id=s.subject_id,
                            recording_id=s.recording_id,
                            is_cough=s.is_cough,
                            similarity_score=s.similarity_score,
                        )
                        for s in p.top_segments
                    ],
                )
                for p in raw_patients
            ]
            return SimilaritySearchResponse(
                query_mode="subject",
                total_results=len(patient_results),
                results=patient_results,
            )
        else:
            # Segment-level results
            raw_segments = retrieval_service.find_similar_segments(
                embedding=request.embedding,
                k=request.k,
            )
            segment_results = [
                SegmentResult(
                    embedding_idx=s.embedding_idx,
                    segment_id=s.segment_id,
                    subject_id=s.subject_id,
                    recording_id=s.recording_id,
                    is_cough=s.is_cough,
                    similarity_score=s.similarity_score,
                )
                for s in raw_segments
            ]
            return SimilaritySearchResponse(
                query_mode="segment",
                total_results=len(segment_results),
                results=segment_results,
            )

    except retrieval_service.RetrievalServiceError as e:
        logger.error(f"Retrieval service error: {e}")
        raise HTTPException(
            status_code=503,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error in similarity search: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {e}",
        ) from e


# ---------------------------------------------------------------------------
# GET /retrieval/health
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=IndexHealthResponse,
    summary="Index Health Check",
    description="Check whether the TurboVec index is loaded and ready.",
)
async def index_health() -> IndexHealthResponse:
    """Report the status of the embedding index."""
    loaded = retrieval_service.is_index_loaded()
    num_vectors = retrieval_service.get_index_size()

    if loaded:
        return IndexHealthResponse(
            index_loaded=True,
            num_vectors=num_vectors,
            message=f"Index ready with {num_vectors:,} vectors.",
        )
    else:
        return IndexHealthResponse(
            index_loaded=False,
            num_vectors=0,
            message=(
                "Index not loaded. Run the index builder first: "
                "python -m retrieval.vector_store.index_builder"
            ),
        )
