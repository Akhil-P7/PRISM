# PRISM — Pediatric Respiratory Intelligence System

## Overview
PRISM is an end‑to‑end AI‑powered platform for pediatric respiratory sound analysis.  It ingests raw audio recordings from children, extracts high‑resolution acoustic features, runs specialised deep‑learning models, stores learned embeddings, and retrieves clinically‑relevant cases to generate diagnostic insights.  The system is built for research agility **and** production robustness.

---

## High‑Level Architecture

![PRISM Architecture Diagram](file:///C:/Users/Dell/.gemini/antigravity-ide/brain/8231ace3-7d63-4b0e-bbf8-2155ab9b5708/prism_architecture_diagram_1781090250312.png)

### Core Building Blocks
| Layer | Component | Responsibility |
|-------|-----------|----------------|
| **Ingestion** | Audio Processing & Feature Extraction | Load WAV/FLAC files, normalise sampling rate (16 kHz), compute mel‑spectrograms, extract temporal descriptors (zero‑crossing rate, RMS, etc.). |
| **Modeling** | Cough Detection CNN | Binary classification and segmentation of cough events within a recording. |
| | Temporal Transformer | Captures longitudinal patterns across repeated recordings, predicts disease progression. |
| | Embedding Generator | Produces a fixed‑size 512‑dim vector summarising the respiratory signature. |
| **Retrieval** | TurboVec Vector Store | Approximate nearest‑neighbour search over embeddings using Google’s TurboQuant algorithm. |
| **API** | FastAPI (Uvicorn) | Orchestrates pipelines, exposes REST endpoints for inference, retrieval and admin tasks. |
| **Persistence** | PostgreSQL | Stores patient metadata, session info, model predictions, audit logs. |
| | TurboVec (disk‑backed) | Stores high‑dimensional embeddings with 4‑bit quantization for rapid similarity search. |
| **Presentation** | Streamlit Dashboard | Interactive UI for clinicians: spectrogram visualisation, inference results, similarity heat‑maps, temporal trend charts. |

---

## Data Flow (Audio → Insight)
1. **Raw Audio** – `.wav`/`.flac` (≥ 10 s, 16 kHz) uploaded via API or UI.
2. **Pre‑processing** – Normalisation → Spectrogram → Feature tensor.
3. **Model Inference** –
   - *Cough Detection* flags cough segments.
   - *Temporal Transformer* analyses sequence of segments.
   - *Embedding Generator* compresses the processed tensor into a 512‑D vector.
4. **Storage** – Embedding saved in TurboVec; metadata & predictions persisted in PostgreSQL.
5. **Retrieval** – TurboVec returns *k* nearest historical embeddings.
6. **Insight Synthesis** – Retrieval results combined with patient history → JSON payload.
7. **Frontend** – Streamlit consumes the JSON, renders spectrograms, similarity cards, and a risk score.

---

## Planned Machine‑Learning Models
| Model | Architecture | Training Data | Output |
|-------|--------------|---------------|--------|
| **Cough Detection CNN** | 2‑D ConvNet (ResNet‑18 backbone) + temporal pooling | COUGHVID, Coswara, ICBHI cough clips (≈ 30 k labelled segments) | Binary mask per frame, confidence score |
| **Temporal Transformer** | Stacked transformer encoder (6 layers, 8 heads) | Longitudinal recordings per patient (≥ 3 sessions) | Progression probability vector (e.g., healthy → mild → severe) |
| **Embedding Generator** | Fully‑connected head on top of the transformer encoder | Same as above, unsupervised contrastive pre‑training on all embeddings | 512‑dim float vector (L2‑normalised) |

All models are implemented in **PyTorch** and exported via `torchscript` for fast inference inside the FastAPI service.

---

## Expected Input / Output
### Input
- **File**: `audio/*.wav` or `audio/*.flac`
- **Sample Rate**: 16 kHz (auto‑resampled if needed)
- **Metadata (JSON)** – `patient_id`, `recording_timestamp`, optional `device_id`

### API Endpoints (excerpt)
```http
POST /api/v1/infer
{
  "patient_id": "P12345",
  "timestamp": "2026-06-09T14:12:00Z",
  "audio_file": "<base64-encoded wav>"
}
```
**Response** (JSON)
```json
{
  "cough_segments": [...],
  "risk_score": 0.73,
  "embedding": [0.12, -0.04, ...],
  "nearest_cases": [
    {"case_id": "C987", "similarity": 0.91},
    {...}
  ]
}
```
The frontend consumes this payload to render visual diagnostics.

---

## Database Design
### PostgreSQL Schema (simplified)
```sql
CREATE TABLE patients (
    patient_id   UUID PRIMARY KEY,
    name        TEXT,
    birth_date  DATE,
    gender      TEXT,
    created_at  TIMESTAMP DEFAULT now()
);

CREATE TABLE recordings (
    recording_id UUID PRIMARY KEY,
    patient_id   UUID REFERENCES patients(patient_id),
    file_path    TEXT,
    duration_sec REAL,
    uploaded_at  TIMESTAMP DEFAULT now()
);

CREATE TABLE cough_events (
    event_id    UUID PRIMARY KEY,
    recording_id UUID REFERENCES recordings(recording_id),
    start_sec   REAL,
    end_sec     REAL,
    confidence  REAL
);

CREATE TABLE model_predictions (
    pred_id      UUID PRIMARY KEY,
    recording_id UUID REFERENCES recordings(recording_id),
    model_name   TEXT,
    result_json  JSONB,
    created_at   TIMESTAMP DEFAULT now()
);
```
### TurboVec Vector Store
- **Index**: `TurboQuantIndex` with `IdMapIndex` wrapper (inner‑product, 4‑bit quantized) stored on disk at `TURBOVEC_INDEX_PATH`.
- **Mapping table** (PostgreSQL) – `embedding_id` ↔ `recording_id` for lookup.

---

## Development Phases (aligned with Bootstrap Guide)
1. **Data Foundation** – adapters for COUGHVID, Coswara, ICBHI; unified schema implementation.
2. **Core Models** – train Cough Detection CNN, prototype Temporal Transformer, generate embeddings.
3. **Retrieval Engine** – build TurboVec index, implement similarity pipeline.
4. **Backend Integration** – FastAPI routes, SQLAlchemy models, Alembic migrations.
5. **Frontend Expansion** – Streamlit dashboards for spectrograms, risk scores, case similarity.
6. **Testing & Evaluation** – unit, integration, pipeline tests; performance benchmarks.
7. **Production Deployment** – Docker‑compose orchestration, CI/CD pipelines, monitoring.

---

## Glossary
- **TurboVec** – Rust vector index with Python bindings, built on Google Research’s TurboQuant algorithm for efficient vector similarity search with 4‑bit quantization.
- **Embedding** – Fixed‑size numeric representation of a high‑dimensional audio signal.
- **Temporal Transformer** – Model that captures sequence dynamics across multiple recordings.
- **Cough Detection CNN** – Convolutional network that isolates cough events from raw audio.

---

*Document generated on 2026‑06‑10. All sections are synchronized with the PRISM Project Bootstrap Guide and reflect the current roadmap.*
