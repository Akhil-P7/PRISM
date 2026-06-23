"""
PRISM Backend — Insights API Router

Endpoints for generating clinical insights via the RATM pipeline.

Routes:
    POST /insights/generate   — Generate insight from patient data
    POST /insights/demo       — Generate demo insight with synthetic data
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from loguru import logger

from backend.schemas.insights import (
    DemoInsightRequest,
    InsightRequest,
    InsightResponse,
    ObservationItem,
)
from backend.services import insight_service

router = APIRouter(prefix="/insights", tags=["Insights"])


def _insight_to_response(insight) -> InsightResponse:
    """Convert a ClinicalInsight to the API response schema."""
    # Build similar cases from the insight's metadata
    # (the insight itself doesn't carry full case data, but the memory does)
    return InsightResponse(
        patient_id=insight.patient_id,
        generated_at=insight.generated_at,
        trajectory_class=insight.trajectory_class,
        trajectory_confidence=insight.trajectory_confidence,
        overall_severity=insight.overall_severity,
        summary=insight.summary,
        observations=[
            ObservationItem(
                category=obs.category,
                severity=obs.severity,
                text=obs.text,
            )
            for obs in insight.observations
        ],
        similar_cases=[],  # populated below if available
        alerts=[],  # populated below if available
        templates_used=insight.templates_used,
        generation_method=insight.generation_method,
    )


# ──────────────────────────────────────────────────────────────────
# POST /insights/generate
# ──────────────────────────────────────────────────────────────────


@router.post(
    "/generate",
    response_model=InsightResponse,
    summary="Generate Clinical Insight",
    description=(
        "Generate a clinical insight from patient data using the full "
        "RATM pipeline (trajectory prediction + retrieval + narrative generation)."
    ),
)
async def generate_insight(request: InsightRequest) -> InsightResponse:
    """Generate a clinical insight from patient data."""
    try:
        insight = insight_service.generate_insight(
            patient_stats=request.patient_stats,
            patient_embedding=request.patient_embedding,
            subject_id=request.subject_id,
            k=request.k,
        )
        return _insight_to_response(insight)

    except insight_service.InsightServiceError as e:
        logger.error(f"Insight service error: {e}")
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Unexpected error generating insight: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}") from e


# ──────────────────────────────────────────────────────────────────
# POST /insights/demo
# ──────────────────────────────────────────────────────────────────


@router.post(
    "/demo",
    response_model=InsightResponse,
    summary="Generate Demo Insight",
    description=(
        "Generate a demo clinical insight using synthetic data. "
        "Useful for testing the dashboard and exploring output format."
    ),
)
async def generate_demo_insight(
    request: DemoInsightRequest | None = None,
) -> InsightResponse:
    """Generate a demo insight with synthetic data."""
    if request is None:
        request = DemoInsightRequest()

    try:
        insight = insight_service.generate_demo_insight(
            trajectory_class=request.trajectory_class,
            subject_id=request.subject_id,
        )
        return _insight_to_response(insight)

    except insight_service.InsightServiceError as e:
        logger.error(f"Demo insight error: {e}")
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Unexpected error in demo insight: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}") from e
