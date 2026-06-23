"""
PRISM Backend — Retrieval Service

Service layer that wraps the TurboVec search engine.  Provides a
singleton pattern so the index is loaded once and shared across
all API requests.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from retrieval.vector_store.search import (
    PatientMatch,
    SearchResult,
    TurboVecSearchEngine,
)

# ---------------------------------------------------------------------------
# Singleton search engine
# ---------------------------------------------------------------------------

_engine: TurboVecSearchEngine | None = None
_lock = threading.Lock()

# Default paths (relative to project root)
DEFAULT_INDEX_PATH = Path("retrieval/vector_store/cough_embeddings.tq")
DEFAULT_METADATA_PATH = Path("retrieval/vector_store/index_metadata.csv")


class RetrievalServiceError(Exception):
    """Raised when the retrieval service encounters an error."""


def get_engine(
    index_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> TurboVecSearchEngine:
    """
    Get or initialise the singleton search engine.

    Thread-safe: only the first caller performs initialisation.

    Args:
        index_path: override path to the .tq index file.
        metadata_path: override path to the metadata CSV.

    Returns:
        The initialised TurboVecSearchEngine.

    Raises:
        RetrievalServiceError: if the index files are not found.
    """
    global _engine

    if _engine is not None:
        return _engine

    with _lock:
        # Double-checked locking
        if _engine is not None:
            return _engine

        idx_path = Path(index_path) if index_path else DEFAULT_INDEX_PATH
        meta_path = Path(metadata_path) if metadata_path else DEFAULT_METADATA_PATH

        if not idx_path.exists():
            raise RetrievalServiceError(
                f"TurboVec index not found at: {idx_path}\n"
                "Run the index builder first:\n"
                "  python -m retrieval.vector_store.index_builder \\\n"
                "    --embeddings-matrix models/embeddings/embeddings_matrix.npy \\\n"
                "    --embeddings-metadata models/embeddings/embeddings_metadata.csv"
            )

        if not meta_path.exists():
            raise RetrievalServiceError(
                f"Index metadata not found at: {meta_path}\n"
                "This file should be created by the index builder."
            )

        _engine = TurboVecSearchEngine(
            index_path=idx_path,
            metadata_path=meta_path,
        )
        return _engine


def is_index_loaded() -> bool:
    """Check whether the search engine has been initialised."""
    return _engine is not None and _engine.is_loaded


def get_index_size() -> int:
    """Return the number of vectors in the loaded index (0 if not loaded)."""
    if _engine is None:
        return 0
    return _engine.num_vectors


# ---------------------------------------------------------------------------
# Search operations
# ---------------------------------------------------------------------------


def find_similar_segments(
    embedding: list[float] | np.ndarray,
    k: int = 10,
) -> list[SearchResult]:
    """
    Find the k most similar segments to a query embedding.

    Args:
        embedding: 512-dim embedding vector.
        k: number of results.

    Returns:
        List of SearchResult.
    """
    engine = get_engine()
    query = np.array(embedding, dtype=np.float32)
    return engine.search(query, k=k)


def find_similar_patients(
    embedding: list[float] | np.ndarray,
    k: int = 5,
) -> list[PatientMatch]:
    """
    Find the k most similar patients (aggregated by subject_id).

    Args:
        embedding: 512-dim embedding vector.
        k: number of unique patients.

    Returns:
        List of PatientMatch.
    """
    engine = get_engine()
    query = np.array(embedding, dtype=np.float32)
    return engine.search_by_subject(query, k=k)
