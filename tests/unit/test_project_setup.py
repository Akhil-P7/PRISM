"""
PRISM — Smoke Test

Validates that the project bootstraps correctly.
"""

import pytest


class TestProjectSetup:
    """Verify the initial project setup is correct."""

    @pytest.mark.unit
    def test_backend_import(self):
        """Backend package should be importable."""
        import backend

        assert hasattr(backend, "__version__")
        assert backend.__version__ == "0.1.0"

    @pytest.mark.unit
    def test_database_import(self):
        """Database package should be importable."""
        import database

        assert database is not None

    @pytest.mark.unit
    def test_models_import(self):
        """Models package should be importable."""
        import models

        assert models is not None

    @pytest.mark.unit
    def test_retrieval_import(self):
        """Retrieval package should be importable."""
        import retrieval

        assert retrieval is not None

    @pytest.mark.unit
    def test_pipelines_import(self):
        """Pipelines package should be importable."""
        import pipelines

        assert pipelines is not None

    @pytest.mark.unit
    def test_evaluation_import(self):
        """Evaluation package should be importable."""
        import evaluation

        assert evaluation is not None
