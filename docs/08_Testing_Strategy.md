# Testing Strategy & Evaluation Framework

# PRISM

## Patient Respiratory Intelligence System

**Version:** 1.0
**Document Type:** Quality Assurance & Testing Specification
**Framework:** Pytest

---

# 1. Purpose

This document outlines the testing methodologies used to ensure the reliability, accuracy, and performance of the PRISM AI pipelines and software infrastructure. Testing in PRISM is divided into two distinct domains:

1. **Software Testing**: Ensuring code correctness, API stability, and database integrity (Unit, Integration, E2E).
2. **AI Evaluation**: Measuring the predictive accuracy and clinical relevance of the machine learning models.

---

# 2. Software Testing Architecture

All software testing is executed via `pytest`. The test suite is organized into the following directories:

```text
tests/
├── unit/            # Isolated function/class tests
├── integration/     # Inter-module communication tests
├── e2e/             # End-to-end API and pipeline tests
└── fixtures/        # Mock audio files, dummy databases, and sample embeddings
```

## 2.1 Unit Testing
**Goal**: Verify that individual functions and classes work in isolation.
* **Backend API**: Testing FastAPI route handlers using `TestClient`. Ensures proper JSON serialization, error handling (e.g., 404s for missing IDs), and status codes.
* **Database**: Testing SQLAlchemy CRUD operations using an in-memory SQLite database (`sqlite:///:memory:`).
* **AI Utilities**: Testing audio normalization scripts, spectrogram generation math, and dataset mapping logic.

## 2.2 Integration Testing
**Goal**: Verify that distinct modules communicate correctly.
* **Retrieval Pipeline**: Ensuring the RATM engine successfully takes a dummy acoustic embedding, queries the TurboVec instance, and retrieves the correct metadata.
* **Database-to-API**: Verifying that records inserted via SQLAlchemy are correctly formatted when queried through the FastAPI endpoints.

## 2.3 End-to-End (E2E) Testing
**Goal**: Validate complete user workflows.
* **Ingestion to Inference**: A simulated test that uploads a dummy `.wav` file, runs it through a mock ResNet model, passes the output to the Temporal Transformer, queries TurboVec, and returns a final Clinical Insight JSON.

---

# 3. AI Evaluation Framework

Software tests ensure the code *runs*; AI evaluation ensures the code is *correct*.

Model evaluations are tracked in the `evaluation/` directory and use standard scikit-learn metrics.

## 3.1 Cough Detection (ResNet-18)
The audio CNN is evaluated on hold-out validation sets from COUGHVID and Coswara.
* **Primary Metrics**:
  * **F1 Score**: Balances precision and recall to handle imbalanced datasets (non-coughs usually outnumber coughs).
  * **ROC-AUC**: Evaluates the model's ability to distinguish between cough and background noise at various thresholds.
  * **Recall**: Prioritized slightly over precision to minimize false negatives (missing a critical respiratory event).

## 3.2 Temporal Intelligence (Transformer)
Evaluates the model's ability to classify 30-day trajectories.
* **Primary Metrics**:
  * **Multi-class Accuracy**: Correctly predicting Stable vs. Increasing vs. Improving trends.
  * **Sequence Log-Loss**: Penalizing high-confidence incorrect trend predictions.

## 3.3 RATM Retrieval (TurboVec)
Evaluates whether the retrieved historical cases are clinically relevant to the current patient.
* **Primary Metrics**:
  * **Precision@K (e.g., P@3)**: What percentage of the top-3 retrieved cases share the same ground-truth diagnosis as the query case.
  * **Recall@K**: Whether the relevant cases in the database were successfully surfaced in the top K results.

---

# 4. Continuous Integration (CI/CD)

PRISM utilizes GitHub Actions to automate testing and evaluation.

1. **Linting & Formatting**: Enforced via `Ruff` and `pre-commit` hooks.
2. **Automated Testing**: On every Pull Request to `main` or `develop`, a GitHub Action spins up a Python environment, installs dependencies via Poetry/Pip, and runs `pytest`.
3. **Model Checks**: The CI pipeline ensures that `models/checkpoints/*.pt` and `models/embeddings/*.csv` are tracked via Git LFS and properly synchronized during deployments.

---

# 5. Mocking & Test Data

To prevent tests from requiring massive datasets or GPU instances, the `tests/fixtures/` directory provides:
* **Mock Audio**: 1-second sine-wave `.wav` files to test audio pipelines without relying on real patient data.
* **Mock Embeddings**: Randomly generated 512-D float vectors to test TurboVec insertion and search logic.
* **Mock Checkpoints**: `unittest.mock` is used to bypass actual PyTorch `forward()` passes during standard software tests, replacing inference with deterministic dummy outputs.
