# PRISM Project Bootstrap Guide

# Pediatric Respiratory Intelligence System

Version: 1.0

Document Type: Development Environment & Project Bootstrap Guide

Status: Pre-Development

---

# 1. Purpose

This document defines the initial setup required for all PRISM contributors.

It standardizes:

* Development environment
* Package management
* Dependency installation
* Code quality standards
* Git workflow
* Local development setup
* Docker configuration
* Project initialization

The goal is to ensure that every developer works within the same reproducible environment.

---

# 2. Development Philosophy

PRISM follows a research-first engineering workflow.

Principles:

* Reproducibility
* Modularity
* Testability
* Scalability
* Maintainability

No code should be merged without:

* Formatting
* Linting
* Type checking
* Testing

---

# 3. Technology Stack

## Core Language

Python 3.11

Reason:

* Stable
* Widely supported
* Excellent ML ecosystem

---

## Backend

FastAPI

Reason:

* Async support
* Automatic OpenAPI generation
* Strong typing

---

## Machine Learning

PyTorch

Reason:

* Research friendly
* Transformer ecosystem
* Strong audio tooling

---

## Audio Processing

Librosa

Torchaudio

---

## Database

PostgreSQL

---

## Vector Store

FAISS

---

## Frontend

Phase 1:

Streamlit

Phase 2:

Next.js

---

# 4. Dependency Management

## Decision

Poetry

Reason:

* Lock files
* Dependency isolation
* Reproducible builds

---

Installation

```bash
pip install poetry
```

Verify

```bash
poetry --version
```

---

# 5. Repository Initialization

Clone repository

```bash
git clone <repo-url>
```

Move into project

```bash
cd PRISM
```

Install dependencies

```bash
poetry install
```

Activate shell

```bash
poetry shell
```

---

# 6. Initial Repository Structure

```text
PRISM/

├── docs/
├── datasets/
├── backend/
├── frontend/
├── models/
├── retrieval/
├── pipelines/
├── database/
├── evaluation/
├── tests/
├── scripts/
├── configs/
├── notebooks/
├── deployment/
│
├── pyproject.toml
├── README.md
├── .gitignore
├── .env.example
└── docker-compose.yml
```

---

# 7. Environment Variables

Create:

```text
.env
```

Example:

```env
DATABASE_URL=postgresql://user:password@localhost/prism

FAISS_INDEX_PATH=./vector_store

DATASET_PATH=./datasets

MODEL_PATH=./models

LOG_LEVEL=INFO
```

Never commit:

```text
.env
```

to Git.

---

# 8. Python Coding Standards

## Formatter

Black

Installation:

```bash
poetry add --group dev black
```

Run:

```bash
black .
```

---

## Import Sorting

isort

```bash
poetry add --group dev isort
```

Run:

```bash
isort .
```

---

## Linter

Ruff

```bash
poetry add --group dev ruff
```

Run:

```bash
ruff check .
```

---

## Type Checking

MyPy

```bash
poetry add --group dev mypy
```

Run:

```bash
mypy .
```

---

# 9. Pre-Commit Hooks

Purpose:

Prevent poor-quality code from entering the repository.

---

Install

```bash
poetry add --group dev pre-commit
```

Initialize

```bash
pre-commit install
```

---

Checks

Before every commit:

```text
Black

isort

Ruff

MyPy
```

---

# 10. Testing Framework

Decision:

Pytest

Install

```bash
poetry add --group dev pytest
```

Run

```bash
pytest
```

---

Structure

```text
tests/

├── unit/
├── integration/
├── pipeline/
└── api/
```

---

# 11. Database Setup

PostgreSQL

Connection String

```env
DATABASE_URL=postgresql://user:password@localhost/prism
```

---

# 12. Docker Strategy

Purpose:

Provide reproducible environments.

---

Dockerfile

Responsible for:

```text
Python

Dependencies

Application Runtime
```

---

docker-compose

Responsible for:

```text
Backend

Database

Vector Store
```

---

Run

```bash
docker compose up
```

---

# 13. Logging Standards

Use:

```python
import logging
```

Never use:

```python
print()
```

for production logic.

---

Log Levels

```text
DEBUG

INFO

WARNING

ERROR

CRITICAL
```

---

# 14. Git Workflow

---

Branches

```text
main

develop

feature/audio

feature/temporal

feature/retrieval

feature/frontend
```

---

Rules

Never commit directly to:

```text
main
```

---

Workflow

```text
Feature Branch
      ↓
Pull Request
      ↓
Code Review
      ↓
Merge Into Develop
      ↓
Release To Main
```

---

# 15. Development Milestone Setup

Before coding begins:

Complete:

### Environment

* Python installed
* Poetry installed

---

### Quality Tools

* Black
* Ruff
* MyPy
* Pytest

---

### Repository

* Clone complete
* Branch strategy established

---

### Documentation

* Architecture reviewed
* Database reviewed
* API reviewed

---

# 16. First Engineering Sprint

Sprint Goal:

Build the Data Foundation Layer

Duration:

1–2 Weeks

Tasks:

### Dataset Setup

* Download COUGHVID
* Download Coswara
* Download ICBHI

---

### Data Adapters

Implement:

```text
coughvid_adapter.py

coswara_adapter.py

icbhi_adapter.py
```

---

### Unified Schema

Implement:

```text
UnifiedSubject

UnifiedRecording

UnifiedSegment
```

---

### Validation

Verify:

```text
Raw Dataset
      ↓
Unified Schema
```

works correctly.

---

# 17. Definition of Done

A task is complete only if:

✓ Code compiles

✓ Tests pass

✓ Documentation updated

✓ Linting passes

✓ Type checking passes

✓ Pull request approved

---

# 18. Bootstrap Checklist

Every contributor must complete:

```text
□ Install Python 3.11

□ Install Poetry

□ Clone Repository

□ Install Dependencies

□ Setup Environment Variables

□ Install Pre-Commit Hooks

□ Run Test Suite

□ Read Architecture Documents

□ Create Feature Branch
```
