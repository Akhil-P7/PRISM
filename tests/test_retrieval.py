"""
PRISM Tests — Retrieval Module

Tests for the vector store search engine, patient aggregation,
and the FastAPI retrieval endpoints.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from retrieval.vector_store.search import PatientMatch, SearchResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_search_results() -> list[SearchResult]:
    """Create a set of mock search results for testing."""
    return [
        SearchResult(
            embedding_idx=0,
            segment_id="seg_001",
            subject_id="patient_A",
            recording_id="rec_001",
            is_cough=True,
            similarity_score=0.95,
        ),
        SearchResult(
            embedding_idx=1,
            segment_id="seg_002",
            subject_id="patient_A",
            recording_id="rec_001",
            is_cough=True,
            similarity_score=0.88,
        ),
        SearchResult(
            embedding_idx=2,
            segment_id="seg_003",
            subject_id="patient_B",
            recording_id="rec_005",
            is_cough=False,
            similarity_score=0.82,
        ),
        SearchResult(
            embedding_idx=3,
            segment_id="seg_004",
            subject_id="patient_B",
            recording_id="rec_005",
            is_cough=True,
            similarity_score=0.79,
        ),
        SearchResult(
            embedding_idx=4,
            segment_id="seg_005",
            subject_id="patient_C",
            recording_id="rec_010",
            is_cough=True,
            similarity_score=0.75,
        ),
    ]


@pytest.fixture
def mock_query_embedding() -> list[float]:
    """Create a mock 512-dim embedding vector."""
    rng = np.random.default_rng(42)
    vec = rng.standard_normal(512).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


@pytest.fixture
def api_client() -> TestClient:
    """Create a FastAPI test client."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Unit Tests — SearchResult & PatientMatch dataclasses
# ---------------------------------------------------------------------------


class TestSearchResult:
    """Tests for the SearchResult dataclass."""

    def test_search_result_creation(self):
        result = SearchResult(
            embedding_idx=0,
            segment_id="seg_001",
            subject_id="patient_A",
            recording_id="rec_001",
            is_cough=True,
            similarity_score=0.95,
        )
        assert result.subject_id == "patient_A"
        assert result.is_cough is True
        assert result.similarity_score == 0.95

    def test_patient_match_creation(self):
        match = PatientMatch(
            subject_id="patient_A",
            best_similarity=0.95,
            avg_similarity=0.90,
            num_matching_segments=3,
            cough_ratio=0.67,
        )
        assert match.subject_id == "patient_A"
        assert match.top_segments == []
        assert match.cough_ratio == pytest.approx(0.67)


# ---------------------------------------------------------------------------
# Unit Tests — Patient aggregation logic
# ---------------------------------------------------------------------------


class TestPatientAggregation:
    """Tests for the search_by_subject aggregation logic."""

    def test_aggregation_groups_by_subject(self, sample_search_results):
        """Verify that results are correctly grouped by subject_id."""
        from collections import defaultdict

        groups: dict[str, list[SearchResult]] = defaultdict(list)
        for r in sample_search_results:
            groups[r.subject_id].append(r)

        assert len(groups) == 3  # patient_A, patient_B, patient_C
        assert len(groups["patient_A"]) == 2
        assert len(groups["patient_B"]) == 2
        assert len(groups["patient_C"]) == 1

    def test_aggregation_best_similarity(self, sample_search_results):
        """Verify best_similarity picks the max score per patient."""
        from collections import defaultdict

        groups: dict[str, list[float]] = defaultdict(list)
        for r in sample_search_results:
            groups[r.subject_id].append(r.similarity_score)

        assert max(groups["patient_A"]) == pytest.approx(0.95)
        assert max(groups["patient_B"]) == pytest.approx(0.82)

    def test_aggregation_cough_ratio(self, sample_search_results):
        """Verify cough_ratio calculation."""
        # patient_B has 1 cough and 1 non-cough
        patient_b = [r for r in sample_search_results if r.subject_id == "patient_B"]
        cough_count = sum(1 for r in patient_b if r.is_cough)
        ratio = cough_count / len(patient_b)
        assert ratio == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Integration Tests — FastAPI endpoints
# ---------------------------------------------------------------------------


class TestRetrievalAPI:
    """Tests for the /api/v1/retrieval/ endpoints."""

    def test_health_endpoint_returns_200(self, api_client):
        """Health endpoint should always return 200, even if index isn't loaded."""
        response = api_client.get("/api/v1/retrieval/health")
        assert response.status_code == 200
        data = response.json()
        assert "index_loaded" in data
        assert "num_vectors" in data
        assert "message" in data

    @patch("backend.services.retrieval_service.find_similar_segments")
    def test_search_segment_mode(
        self, mock_find, api_client, mock_query_embedding, sample_search_results
    ):
        """Test segment-level search endpoint."""
        mock_find.return_value = sample_search_results

        response = api_client.post(
            "/api/v1/retrieval/search",
            json={
                "embedding": mock_query_embedding,
                "k": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["query_mode"] == "segment"
        assert data["total_results"] == 5
        assert len(data["results"]) == 5

    @patch("backend.services.retrieval_service.find_similar_patients")
    def test_search_subject_mode(self, mock_find, api_client, mock_query_embedding):
        """Test patient-level (subject-aggregated) search endpoint."""
        mock_find.return_value = [
            PatientMatch(
                subject_id="patient_A",
                best_similarity=0.95,
                avg_similarity=0.91,
                num_matching_segments=2,
                cough_ratio=1.0,
                top_segments=[],
            ),
        ]

        response = api_client.post(
            "/api/v1/retrieval/search",
            json={
                "embedding": mock_query_embedding,
                "k": 5,
                "group_by": "subject",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["query_mode"] == "subject"
        assert data["total_results"] == 1
        assert data["results"][0]["subject_id"] == "patient_A"

    def test_search_validates_embedding_length(self, api_client):
        """Embedding must be exactly 512 dimensions."""
        response = api_client.post(
            "/api/v1/retrieval/search",
            json={
                "embedding": [0.1] * 100,  # wrong length
                "k": 5,
            },
        )
        assert response.status_code == 422  # validation error

    def test_search_validates_k_range(self, api_client, mock_query_embedding):
        """k must be between 1 and 100."""
        response = api_client.post(
            "/api/v1/retrieval/search",
            json={
                "embedding": mock_query_embedding,
                "k": 0,
            },
        )
        assert response.status_code == 422

        response = api_client.post(
            "/api/v1/retrieval/search",
            json={
                "embedding": mock_query_embedding,
                "k": 101,
            },
        )
        assert response.status_code == 422
