# 01 - PRISM: System Overview & Bootstrap Guide

**Version:** 1.0
**Document Type:** System Overview & Engineering Onboarding

---

## 1. Executive Overview

PRISM (Patient Respiratory Intelligence System) is an end‑to‑end AI‑powered platform for Patient respiratory sound analysis. It ingests raw audio recordings from patients, extracts high‑resolution acoustic features, runs specialized deep‑learning models, stores learned embeddings, and retrieves clinically‑relevant cases to generate diagnostic insights.

This document serves as both the high-level architecture overview and the mandatory onboarding guide for all PRISM engineering contributors.

---

## 2. High‑Level Architecture

![PRISM Architecture Diagram](file:///C:/Users/Dell/.gemini/antigravity-ide/brain/8231ace3-7d63-4b0e-bbf8-2155ab9b5708/prism_architecture_diagram_1781090250312.png)

### Core Building Blocks
| Layer | Component | Responsibility |
|-------|-----------|----------------|
| **Ingestion** | Audio Processing & Feature Extraction | Load WAV/FLAC files, normalize sampling rate (16 kHz), compute mel‑spectrograms, extract temporal descriptors. |
| **Modeling** | Cough Detection CNN | Binary classification and segmentation of cough events within a recording. |
| | Temporal Transformer | Captures longitudinal patterns across repeated recordings, predicts disease progression. |
| | Embedding Generator | Produces a fixed‑size 512‑dim vector summarizing the respiratory signature. |
| **Retrieval** | TurboVec Vector Store | Approximate nearest‑neighbour search over embeddings using Google’s TurboQuant algorithm. |
| **API** | FastAPI (Uvicorn) | Orchestrates pipelines, exposes REST endpoints for inference, retrieval, and admin tasks. |
| **Persistence** | PostgreSQL | Stores patient metadata, session info, model predictions, audit logs. |
| | TurboVec (disk‑backed) | Stores high‑dimensional embeddings with 4‑bit quantization for rapid similarity search. |
| **Presentation** | Streamlit Dashboard | Interactive UI for clinicians: spectrogram visualization, inference results, similarity heat‑maps, temporal trend charts. |

---

## 3. Data Flow (Audio → Insight)

1. **Raw Audio** – `.wav`/`.flac` (≥ 10 s, 16 kHz) uploaded via API or UI.
2. **Pre‑processing** – Normalization → Spectrogram → Feature tensor.
3. **Model Inference** –
   - *Cough Detection* flags cough segments.
   - *Temporal Transformer* analyzes sequence of segments.
   - *Embedding Generator* compresses the processed tensor into a 512‑D vector.
4. **Storage** – Embedding saved in TurboVec; metadata & predictions persisted in PostgreSQL.
5. **Retrieval** – TurboVec returns *k* nearest historical embeddings.
6. **Insight Synthesis** – Retrieval results combined with patient history → JSON payload.
7. **Frontend** – Streamlit consumes the JSON, renders visual diagnostics and a risk score.

---

## 4. Machine‑Learning Models (V1.0 Completed)

All AI models in PRISM have been fully implemented in **PyTorch** and are exported via `torchscript` for fast inference inside the FastAPI service.

| Model | Architecture | Training Data | Output |
|-------|--------------|---------------|--------|
| **Cough Detection CNN** | 2‑D ConvNet (ResNet‑18 backbone) | COUGHVID, Coswara, ICBHI (≈131k segments) | Binary mask, confidence score |
| **Temporal Transformer** | Stacked transformer encoder (3 layers, 4 heads) | Longitudinal synthetic patient trajectories | Progression trajectory (Stable, Improving, Increasing, Abnormal) |
| **Embedding Generator** | L2-Normalized projection head on the CNN | Same as above | 512‑dim float vector |

---

## 5. Engineering Standards & Bootstrap Guide

PRISM follows a research-first engineering workflow prioritizing reproducibility, modularity, and testability.

### 5.1 Technology Stack
- **Core Language:** Python 3.11 (Stable, excellent ML ecosystem)
- **Backend:** FastAPI (Async support, strong typing)
- **Machine Learning:** PyTorch, Librosa, Torchaudio
- **Database:** PostgreSQL & TurboVec
- **Frontend:** Streamlit (Phase 1)

### 5.2 Dependency Management (Poetry)
We exclusively use Poetry for lock files, dependency isolation, and reproducible builds.
```bash
pip install poetry
poetry install
poetry shell
```

### 5.3 Python Coding Standards & Pre-Commit
No code should be merged without passing our automated quality gates.

- **Formatter:** `Black`
- **Import Sorting:** `isort`
- **Linter:** `Ruff`
- **Type Checking:** `MyPy`

Initialize your environment before your first commit:
```bash
poetry add --group dev pre-commit
pre-commit install
```

### 5.4 Testing Framework
We use **Pytest**. Tests are divided into `tests/unit/`, `tests/integration/`, and `tests/e2e/`.
```bash
pytest
```

### 5.5 Git Workflow
- Never commit directly to `main`.
- Branching strategy: `feature/audio`, `feature/temporal`, `feature/frontend`.
- Workflow: Feature Branch -> Pull Request -> Code Review -> Merge into `develop` -> Release to `main`.

### 5.6 Environment Setup
Create a `.env` file at the root of the project (never commit this to Git).
```env
DATABASE_URL=postgresql://user:password@localhost/prism
TURBOVEC_INDEX_PATH=./vector_store
DATASET_PATH=./datasets
MODEL_PATH=./models
LOG_LEVEL=INFO
```
