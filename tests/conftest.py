"""
PRISM Tests — Shared Fixtures

Common test fixtures and configuration for the PRISM test suite.
"""

import pytest


@pytest.fixture
def sample_audio_config():
    """Provide default audio processing configuration for tests."""
    return {
        "sample_rate": 44100,
        "channels": 1,
        "segment_duration": 3.0,
        "n_mels": 128,
        "n_fft": 2048,
        "hop_length": 512,
    }


@pytest.fixture
def sample_subject_data():
    """Provide sample subject data for tests."""
    return {
        "source_subject_id": "test_subject_001",
        "age": 8,
        "age_group": "Child",
        "gender": "Male",
        "country": "India",
        "health_status": "Healthy",
    }


@pytest.fixture
def sample_recording_data():
    """Provide sample recording data for tests."""
    return {
        "file_path": "datasets/raw/coughvid/sample.wav",
        "duration": 5.2,
        "sample_rate": 44100,
        "channels": 1,
        "recording_type": "Cough",
    }
