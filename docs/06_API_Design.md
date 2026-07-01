# API Design Document

# PRISM

## Patient Respiratory Intelligence System

**Version:** 1.0
**Document Type:** API Specification
**Architecture Style:** REST API
**Framework:** FastAPI
**Data Format:** JSON

---

# 1. Purpose

This document defines the communication layer between the Frontend Dashboard, AI Services, Database, and Retrieval Engine. The API follows a service-oriented architecture.

---

# 2. API Architecture

```text
Frontend
     │
     ▼
 FastAPI Gateway
     │
 ┌───┴───────────┐
 ▼               ▼
Retrieval      Insight
   API           API
```

---

# 3. Implemented Services (Version 1.0)

The V1 backend (`backend/main.py`) exposes the core RATM (Retrieval-Augmented Temporal Modeling) functionalities via two primary routers.

---

## 3.1 Retrieval APIs (`/api/v1/retrieval`)

Responsible for querying the TurboVec Vector Database and managing the acoustic embeddings memory store.

### Create Embedding
POST `/api/v1/retrieval/embed`
Generates a 512-D acoustic embedding from an audio file.

### Search Similar Sessions
POST `/api/v1/retrieval/search`
Queries TurboVec to find historically similar patient cases based on the acoustic embedding.
**Response:**
```json
{
  "matches": [
    {
      "memory_id": "uuid",
      "score": 0.91,
      "metadata": {}
    }
  ]
}
```

### Store Memory
POST `/api/v1/retrieval/memory`
Stores a new clinical memory into the database and updates the TurboVec index.

### Get Memory
GET `/api/v1/retrieval/memory/{memory_id}`
Retrieves details of a specific clinical memory.

---

## 3.2 Insight APIs (`/api/v1/insights`)

Responsible for generating explainable clinical observations using the context retrieved from TurboVec.

### Generate Insight
POST `/api/v1/insights/generate`
Generates a human-readable insight report using the patient's temporal trajectory, disease prediction, and retrieved historical cases.
**Response:**
```json
{
  "insight": "Nighttime cough frequency is highly prominent, typical of asthma presentations, and matches 3 historical cases..."
}
```

### Get Insights
GET `/api/v1/insights`
Retrieves a list of previously generated clinical insights.

### Get Insight
GET `/api/v1/insights/{insight_id}`
Retrieves a specific insight by ID.

---

# 4. Future Expansion (V2 API Roadmap)

While V1 runs audio processing, cough detection, and temporal intelligence dynamically within the application pipeline (or via CLI scripts), V2 will expose these as dedicated REST microservices to support external integrations.

### Dataset & Recording APIs
* `GET /api/v2/datasets`
* `POST /api/v2/recordings`

### Audio Processing & Detection APIs
* `POST /api/v2/audio/process` (Generate spectrograms)
* `POST /api/v2/detection/cough` (Trigger ResNet-18 detector)

### Temporal Intelligence APIs
* `POST /api/v2/temporal/analyze` (Trigger Temporal Transformer)
* `GET /api/v2/temporal/{subject_id}` (Get 30-day trajectory)

### Environmental APIs
* `POST /api/v2/environment/correlation` (Calculate AQI/weather impacts)

### Dashboard & Evaluation APIs
* `GET /api/v2/dashboard/overview` (System metrics)
* `GET /api/v2/evaluation/models` (AI performance metrics)

---

# 5. Internal AI Service Flow (Conceptual)

Even when running in a monolithic or script-based context, the internal data flow mirrors the API design:

```text
Recording Uploaded
       │
       ▼
Audio Processor (Mel Spectrogram)
       │
       ▼
ResNet-18 Detection
       │
       ▼
Temporal Transformer
       │
       ▼
Retrieval API (TurboVec)
       │
       ▼
Insight API
```

This REST architecture ensures that PRISM remains modular, scalable, and compatible with future healthcare deployments such as mobile apps, wearable backends, or hospital EMR integrations.
