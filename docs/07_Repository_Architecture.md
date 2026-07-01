# Repository Architecture & Development Roadmap

# PRISM

## Pediatric Respiratory Intelligence System

**Version:** 1.0
**Document Type:** Repository Architecture & Development Roadmap
**Status:** Engineering Blueprint

---

# 1. Purpose

This document defines the repository structure, module ownership, development workflow, engineering standards, and development milestones. The objective is to ensure that every component of PRISM can be developed independently while remaining fully integrable.

---

# 2. Repository Philosophy

PRISM follows a highly modular architecture. Each major subsystem (AI inference, RATM engine, Database, APIs, UI) is decoupled so it can be developed, tested, and deployed independently while sharing common interfaces.

---

# 3. Repository Overview

```text
PRISM/
├── backend/       # FastAPI REST services (Insights & Retrieval APIs)
├── configs/       # Configuration YAMLs
├── database/      # SQLite/PostgreSQL schemas and migrations
├── datasets/      # Data ingestion pipelines and raw/processed storage
├── docs/          # Project documentation (Architecture, API, DB specs)
├── evaluation/    # Model metrics and clinical validation reports
├── frontend/      # Streamlit user interface
├── models/        # AI Models (ResNet-18, Temporal Transformer, Checkpoints)
├── notebooks/     # Jupyter notebooks for EDA and model prototyping
├── pipelines/     # Orchestration scripts for end-to-end processing
├── retrieval/     # RATM Engine (TurboVec, Memory Builder)
├── scripts/       # CLI utilities (Deployment, setup, DB seeding)
└── tests/         # Unit, integration, and E2E test suites
```

---

# 4. Detailed Module Breakdown

## 4.1 AI Module (`models/`)
Contains all PyTorch machine learning models, training scripts, and exported checkpoints.
* **`cough_detector/`**: ResNet-18 CNN for cough classification and 512-D acoustic embedding generation.
* **`temporal_transformer/`**: 30-day temporal dependency modeling to classify trajectories (Stable, Increasing, Abnormal).
* **`disease_classifier/`**: MLP for mapping acoustic embeddings to specific conditions (Asthma, Pneumonia, etc.).
* **`embeddings/`**: `embeddings_metadata.csv` and TurboVec index assets (managed via Git LFS).

## 4.2 Retrieval-Augmented Temporal Modeling (`retrieval/`)
The core reasoning engine of PRISM.
* **Vector Store Interface**: Manages TurboVec indexing and Cosine Similarity search.
* **Memory Builder**: Assembles patient trajectories and retrieved historical embeddings into structured context.
* **Insight Generator**: Uses clinical templates to synthesize explanations for model predictions.

## 4.3 Backend API (`backend/`)
FastAPI application that serves the PRISM AI engine to external clients.
* **`main.py`**: FastAPI entry point configuring CORS, middleware, and routers.
* **`api/insights.py`**: Router for triggering the Clinical Insight Generator.
* **`api/retrieval.py`**: Router for querying TurboVec embeddings.

## 4.4 Data & Database (`datasets/`, `database/`)
* **`datasets/`**: Handlers for COUGHVID, Coswara, and ICBHI parsing. Output feeds into the database.
* **`database/models/`**: SQLAlchemy ORM classes (`Dataset`, `Subject`, `Recording`).

## 4.5 Frontend UI (`frontend/`)
Streamlit-based clinical dashboard.
* **`app.py`**: Main application logic.
* **`pages/`**: Routing for Insights, Patient Overview, and System Analytics.

---

# 5. Git Workflow & Version Control

## Branching Strategy
* **`main`**: Production-ready code. Commits require passing CI/CD pipelines.
* **`develop`**: Active integration branch.
* **`feature/*`**: Scoped branches for new capabilities (e.g., `feature/resnet18-upgrade`).
* **`bugfix/*`**: Patches for identified issues.

## Large File Storage (LFS)
PRISM uses **Git LFS** to version control large binary files.
* **Tracked extensions**: `*.pt` (PyTorch checkpoints), `*.tq` (TurboVec indices).
* **Hugging Face Sync**: Automated GitHub Actions deploy the LFS assets directly to Hugging Face Spaces for production hosting.

---

# 6. Team Responsibilities

* **Data Engineering Team**: Dataset ingestion, data normalization, SQLite migrations.
* **Audio AI Team**: Mel Spectrogram generation, ResNet-18 training, and tuning.
* **Temporal AI Team**: Feature engineering and Transformer development.
* **Retrieval Team**: TurboVec integration, RATM context building, and Clinical Insight prompt engineering.
* **Full-Stack Team**: FastAPI routing, Streamlit dashboard UX, and deployment automation.
