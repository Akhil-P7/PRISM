<div align="center">

# 🫁 PRISM

### Pediatric Respiratory Intelligence System

**A modular AI platform for respiratory sound analysis, temporal intelligence, and retrieval-augmented clinical insight generation.**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## 📋 Overview

PRISM transforms raw respiratory audio into clinically meaningful intelligence through a multi-stage AI pipeline:

```
Raw Audio → Cough Detection → Temporal Analysis → Environmental Correlation → Retrieval-Augmented Insights
```

Unlike simple cough classifiers, PRISM provides **explainable, temporally-aware respiratory monitoring** powered by a novel **Retrieval-Augmented Temporal Modeling (RATM)** architecture.

---

## 🏗️ Architecture

```
                    User
                      │
                      ▼
              Dashboard Interface
                      │
                      ▼
                 Backend API (FastAPI)
                      │
      ┌───────────────┼────────────────┐
      ▼               ▼                ▼
Audio Engine   Temporal Engine   Retrieval Engine
      │               │                │
      └───────────────┼────────────────┘
                      ▼
                Data Storage
          (PostgreSQL + FAISS)
```

### Core Modules

| Module | Description | Technology |
|--------|-------------|------------|
| **Audio Processing Engine** | Audio loading, normalization, segmentation, spectrogram generation | Librosa, Torchaudio |
| **Cough Detection Engine** | CNN-based cough event detection from mel spectrograms | PyTorch |
| **Temporal Intelligence** | Transformer-based trend analysis over cough event sequences | PyTorch Transformer |
| **Environmental Correlation** | AQI, temperature, humidity correlation analysis | SciPy, Pandas |
| **Retrieval-Augmented (RATM)** | Explainable AI via historical memory retrieval | FAISS, LLM Layer |
| **Visualization Layer** | Interactive dashboard with trends, analytics, insights | Streamlit → Next.js |

---

## 📁 Repository Structure

```
PRISM/
├── backend/                # FastAPI backend (API, services, repositories)
│   ├── api/                # Endpoint definitions
│   ├── services/           # Business logic
│   ├── repositories/       # Database interactions
│   ├── schemas/            # Pydantic models
│   ├── middleware/         # Request middleware
│   └── main.py            # Application entrypoint
│
├── models/                 # Machine learning models
│   ├── cough_detector/     # CNN cough detection
│   ├── temporal_transformer/ # Temporal trend analysis
│   ├── embeddings/         # Embedding generation
│   └── shared/             # Shared model utilities
│
├── retrieval/              # Retrieval-Augmented Intelligence
│   ├── embeddings/         # Respiratory embeddings
│   ├── vector_store/       # FAISS index management
│   ├── retrieval_engine/   # Similarity search
│   ├── memory_builder/     # Memory object construction
│   └── insight_generator/  # Clinical insight generation
│
├── pipelines/              # End-to-end workflow pipelines
│   ├── ingestion/          # Dataset ingestion
│   ├── preprocessing/      # Audio preprocessing
│   ├── detection/          # Cough detection pipeline
│   ├── temporal/           # Temporal analysis pipeline
│   ├── retrieval/          # Retrieval pipeline
│   └── orchestration/      # Pipeline orchestration
│
├── database/               # Database layer
│   ├── models/             # SQLAlchemy ORM models
│   ├── migrations/         # Alembic migrations
│   ├── seeds/              # Seed data
│   └── connection.py       # DB connection management
│
├── datasets/               # Dataset storage
│   ├── raw/                # Raw datasets (COUGHVID, Coswara, ICBHI)
│   ├── processed/          # Processed & unified data
│   ├── features/           # Extracted features cache
│   └── metadata/           # Dataset metadata
│
├── evaluation/             # Model evaluation framework
│   ├── cough_detection/    # Detection metrics
│   ├── temporal_analysis/  # Temporal metrics
│   ├── retrieval/          # Retrieval metrics
│   └── reports/            # Evaluation reports
│
├── frontend/               # Dashboard (Streamlit Phase 1)
│   ├── pages/              # Dashboard pages
│   ├── components/         # Reusable UI components
│   ├── charts/             # Visualization components
│   └── app.py              # Streamlit entrypoint
│
├── tests/                  # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   ├── pipeline/           # End-to-end pipeline tests
│   └── api/                # API endpoint tests
│
├── configs/                # Configuration files
│   ├── database.yaml       # Database configuration
│   ├── training.yaml       # Model training configuration
│   ├── inference.yaml      # Inference configuration
│   └── retrieval.yaml      # Retrieval configuration
│
├── scripts/                # Utility scripts
├── notebooks/              # Jupyter notebooks for research
├── deployment/             # Docker & deployment configs
├── docs/                   # Project documentation
│
├── pyproject.toml          # Project config & dependencies
├── docker-compose.yml      # Multi-service Docker setup
├── Dockerfile              # Application container
├── .pre-commit-config.yaml # Pre-commit hooks
├── .env.example            # Environment variable template
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **Poetry** (dependency management)
- **Docker & Docker Compose** (optional, for containerized development)
- **PostgreSQL** (production) or SQLite (development)

### 1. Clone the Repository

```bash
git clone <repo-url>
cd PRISM
```

### 2. Install Dependencies

```bash
pip install poetry
poetry install
```

### 3. Activate Environment

```bash
poetry shell
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your local configuration
```

### 5. Install Pre-Commit Hooks

```bash
pre-commit install
```

### 6. Run Tests

```bash
pytest
```

### 7. Start the Backend API

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 8. Start the Dashboard (Streamlit)

```bash
streamlit run frontend/app.py
```

---

## 🐳 Docker Quick Start

```bash
docker compose up --build
```

This starts:
- **PRISM Backend API** on port `8000`
- **PostgreSQL Database** on port `5432`
- **Streamlit Dashboard** on port `8501`

---

## 📊 Datasets

PRISM integrates three respiratory sound datasets:

| Dataset | Role | Used For |
|---------|------|----------|
| **COUGHVID V3** | Primary | CNN cough detector training, feature extraction |
| **Coswara** | Secondary | Validation, robustness testing |
| **ICBHI 2017** | Auxiliary | Respiratory pattern analysis, generalization |

All datasets are normalized into a **Unified Data Model** through dedicated adapters.

---

## 🧪 Development Standards

| Tool | Purpose | Command |
|------|---------|---------|
| **Black** | Code formatting | `black .` |
| **isort** | Import sorting | `isort .` |
| **Ruff** | Linting | `ruff check .` |
| **MyPy** | Type checking | `mypy .` |
| **Pytest** | Testing | `pytest` |

All checks run automatically via **pre-commit hooks** before every commit.

---

## 🌿 Git Workflow

```
Feature Branch → Pull Request → Code Review → Merge into develop → Release to main
```

- **Never commit directly to `main`**
- All features use **pull requests**
- **Code review required** before merge

### Branch Naming

```
main                    # Production releases
develop                 # Integration branch
feature/audio           # Audio processing features
feature/temporal        # Temporal intelligence features
feature/retrieval       # Retrieval engine features
feature/frontend        # Dashboard features
```

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [System Architecture](docs/02_System_Architecture.md) | Overall platform architecture |
| [AI Model Design](docs/03_AI_Model_Design.md) | ML pipeline & RATM architecture |
| [Database Design](docs/04_Database_Design.md) | Schema & data model |
| [Data Ingestion](docs/05_Data_Ingestion_Model.md) | Dataset integration & adapters |
| [API Design](docs/06_API_Design.md) | REST API specification |
| [Repository Architecture](docs/07_Repository_Architecture.md) | Code structure & roadmap |
| [Testing Strategy](docs/08_Testing_Strategy.md) | Testing framework & approach |
| [Deployment Guide](docs/09_Deployment_Guide.md) | Docker & deployment |

---

## 📅 Development Roadmap

| Phase | Focus | Duration |
|-------|-------|----------|
| **Phase 1** | Data Foundation — Dataset setup, unified schema, ingestion pipeline | 2 weeks |
| **Phase 2** | Audio Intelligence — Spectrogram generation, CNN detector | 3 weeks |
| **Phase 3** | Temporal Intelligence — Event generation, Transformer development | 3 weeks |
| **Phase 4** | Retrieval Intelligence — Embedding generation, FAISS integration | 2 weeks |
| **Phase 5** | Backend Development — FastAPI, database integration | 2 weeks |
| **Phase 6** | Dashboard Development — Streamlit interface | 1 week |
| **Phase 7** | Integration & Testing — End-to-end validation | 2 weeks |

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ by the PRISM Team**

*Advancing pediatric respiratory health through explainable AI*

</div>
