---
title: PRISM - Patient Respiratory Intelligence
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.58.0
app_file: frontend/app.py
pinned: false
---

<div align="center">

# 🫁 PRISM

### Patient Respiratory Intelligence System

**A state-of-the-art AI platform for Patient respiratory analysis, longitudinal temporal intelligence, and retrieval-augmented clinical diagnostics.**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Deployed-F9AB00?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## 📋 Overview

**Version 1.0 - Production Ready**

PRISM transforms raw respiratory audio into clinically meaningful intelligence. Unlike standard binary classifiers, PRISM provides **explainable, temporally-aware respiratory monitoring** powered by a proprietary **Retrieval-Augmented Temporal Modeling (RATM)** architecture.

It answers not just *"Is this a cough?"*, but critically: *"Is the patient improving or deteriorating compared to last week, and what similar historical cases support this diagnosis?"*

```text
Raw Audio → CNN Cough Detection → Transformer Temporal Analysis → TurboVec Similarity Search → Clinical Insight
```

---

## 🏗️ Architecture & Core Intelligence

PRISM utilizes a highly modular, decoupled architecture where heavy GPU inference and rapid CPU vector search work in tandem:

| Module | Purpose | Technology |
|--------|-------------|------------|
| **Cough Detection CNN** | A highly optimized ResNet-18 backbone adapted for 1-channel Mel-Spectrograms. Detects cough events and extracts deep 512-D acoustic features. | PyTorch |
| **Temporal Transformer** | A 3-layer encoder-only Transformer that analyzes 30-day longitudinal cough statistics to predict disease trajectory (Stable, Improving, Increasing, Abnormal). | PyTorch |
| **Retrieval Engine (RATM)** | Ultra-fast semantic search across 131k+ historical respiratory embeddings using 4-bit quantization, retrieving clinically similar cases in milliseconds. | TurboVec |
| **Unified Data Foundation** | Standardizes disparate datasets (COUGHVID, Coswara, ICBHI) into a clean, relational schema. | PostgreSQL / SQLAlchemy |
| **Presentation & API** | Interactive clinical dashboard backed by a robust asynchronous API. | Streamlit & FastAPI |

---

## 📚 Comprehensive Documentation

The PRISM architecture, design decisions, and engineering standards are meticulously documented. New developers should start with document `01`.

| Document | Description |
|----------|-------------|
| **[01_System_Overview_and_Bootstrap](docs/01_System_Overview_and_Bootstrap.md)** | Start Here: Executive summary and engineering onboarding. |
| **[02_System_Architecture](docs/02_System_Architecture.md)** | Data flow and pipeline design. |
| **[03_AI_Model_Design](docs/03_AI_Model_Design.md)** | Detailed ML architecture (CNN, Transformer, TurboVec). |
| **[04_Database_Design](docs/04_Database_Design.md)** | PostgreSQL schema and SQLAlchemy mapping. |
| **[05_Data_Ingestion_Model](docs/05_Data_Ingestion_Model.md)** | Dataset adapters and normalisation logic. |
| **[06_API_Design](docs/06_API_Design.md)** | FastAPI REST endpoint specifications. |
| **[07_Repository_Architecture](docs/07_Repository_Architecture.md)** | Codebase structure and Git LFS tracking. |
| **[08_Testing_Strategy](docs/08_Testing_Strategy.md)** | Pytest framework and AI evaluation metrics. |
| **[09_Deployment_Guide](docs/09_Deployment_Guide.md)** | GitHub Actions CI/CD to Hugging Face Spaces. |

> **Note:** Deep-dive technical sprint post-mortems (Colab workflows, hyperparameter tuning, synthetic data math) can be found in `docs/technical_deep_dives/`.

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python 3.11+**
- **Poetry** (Strict dependency management)
- **Git LFS** (Required for pulling `.pt` models and TurboVec indices)

### Local Installation
```bash
# 1. Clone the repository (Ensure Git LFS is installed to pull model weights)
git clone <repo-url>
cd PRISM
git lfs fetch --all origin

# 2. Install dependencies via Poetry
pip install poetry
poetry install
poetry shell

# 3. Configure your environment
cp .env.example .env

# 4. Start the FastAPI Backend
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# 5. Start the Streamlit Clinical Dashboard
streamlit run frontend/app.py
```

### Docker Deployment
For rapid containerized deployment:
```bash
docker compose up --build
```

---

## 🧪 Development & Quality Standards

PRISM enforces strict CI/CD quality gates. No code is merged to `main` without passing:

| Tool | Purpose | Command |
|------|---------|---------|
| **Black** | Code formatting | `black .` |
| **Ruff** | Aggressive linting | `ruff check .` |
| **MyPy** | Strict type checking | `mypy .` |
| **Pytest** | Unit & Integration testing | `pytest` |

*Ensure you install the pre-commit hooks (`pre-commit install`) before submitting a Pull Request.*

---

## ☁️ Hugging Face Deployment

PRISM is automatically deployed to Hugging Face Spaces via GitHub Actions. Any push to the `main` branch will trigger the `huggingface_sync.yml` workflow.

**Critical:** Because PRISM uses Git LFS for PyTorch checkpoints (`.pt`) and TurboVec embeddings (`.csv`), you must ensure your GitHub Actions runner executes `git lfs fetch --all origin` before pushing to the HF remote. See `docs/09_Deployment_Guide.md` for the exact YAML configuration.

---

<div align="center">

**Built with ❤️ by the PRISM Team**

*Advancing Patient respiratory health through explainable, temporal AI.*

</div>
