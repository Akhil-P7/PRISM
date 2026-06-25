# API Design Document

# PRISM

## Pediatric Respiratory Intelligence System

Version: 1.0

Document Type: API Specification

Architecture Style: REST API

Framework: FastAPI

Data Format: JSON

---

# 1. Purpose

This document defines the communication layer between:

* Frontend Dashboard
* AI Services
* Database
* Retrieval Engine

The API follows a service-oriented architecture.

---

# 2. API Architecture

```text
Frontend
     │
     ▼
 FastAPI Gateway
     │
 ┌───┼───────────┐
 ▼   ▼           ▼
Data AI      Retrieval
API  API      API
```

---

# 3. Service Categories

The API is divided into:

### Dataset APIs

Manage datasets.

### Recording APIs

Manage audio recordings.

### Event APIs

Manage cough events.

### Analytics APIs

Temporal analysis.

### Retrieval APIs

RATM engine.

### Insight APIs

Clinical explanations.

---

# 4. Dataset APIs

---

## Get Datasets

GET

```http
/api/v1/datasets
```

Response

```json
[
  {
    "id":"1",
    "name":"COUGHVID"
  }
]
```

---

## Get Dataset Details

GET

```http
/api/v1/datasets/{dataset_id}
```

---

# 5. Subject APIs

---

## Get Subjects

GET

```http
/api/v1/subjects
```

---

## Get Subject

GET

```http
/api/v1/subjects/{subject_id}
```

Response

```json
{
  "subject_id":"123",
  "age_group":"Adult",
  "gender":"Male"
}
```

---

# 6. Recording APIs

---

## Upload Recording

POST

```http
/api/v1/recordings
```

Body

```json
{
  "file":"audio.wav"
}
```

---

## Get Recording

GET

```http
/api/v1/recordings/{recording_id}
```

---

## List Recordings

GET

```http
/api/v1/recordings
```

Filters

```text
dataset
subject
date
```

---

# 7. Audio Processing APIs

---

## Process Recording

POST

```http
/api/v1/audio/process
```

Purpose

Generate:

* Spectrogram
* Metadata

Response

```json
{
  "recording_id":"123",
  "status":"processed"
}
```

---

## Generate Spectrogram

POST

```http
/api/v1/audio/spectrogram
```

Response

```json
{
  "spectrogram_path":"..."
}
```

---

# 8. Cough Detection APIs

---

## Detect Events

POST

```http
/api/v1/detection/cough
```

Response

```json
{
  "events_detected":12
}
```

---

## Get Events

GET

```http
/api/v1/events
```

Filters

```text
recording_id
subject_id
```

---

## Event Details

GET

```http
/api/v1/events/{event_id}
```

---

# 9. Temporal Intelligence APIs

---

## Generate Temporal Features

POST

```http
/api/v1/temporal/features
```

Response

```json
{
  "cough_count":24,
  "night_ratio":0.61
}
```

---

## Trend Analysis

POST

```http
/api/v1/temporal/analyze
```

Response

```json
{
  "trend":"Increasing"
}
```

---

## Temporal Summary

GET

```http
/api/v1/temporal/{subject_id}
```

---

# 10. Environmental APIs

---

## Add Environment Data

POST

```http
/api/v1/environment
```

---

## Get Environment Data

GET

```http
/api/v1/environment/{recording_id}
```

---

## Correlation Analysis

POST

```http
/api/v1/environment/correlation
```

Response

```json
{
  "aqi_correlation":0.71
}
```

---

# 11. Retrieval APIs

---

## Create Embedding

POST

```http
/api/v1/retrieval/embed
```

---

## Search Similar Sessions

POST

```http
/api/v1/retrieval/search
```

Response

```json
{
  "matches":[
      {
        "score":0.91
      }
  ]
}
```

---

## Store Memory

POST

```http
/api/v1/retrieval/memory
```

---

## Get Memory

GET

```http
/api/v1/retrieval/memory/{memory_id}
```

---

# 12. Insight APIs

---

## Generate Insight

POST

```http
/api/v1/insights/generate
```

Response

```json
{
  "insight":
  "Night cough burden increased..."
}
```

---

## Get Insights

GET

```http
/api/v1/insights
```

---

## Get Insight

GET

```http
/api/v1/insights/{insight_id}
```

---

# 13. Dashboard APIs

---

## Dashboard Overview

GET

```http
/api/v1/dashboard/overview
```

Response

```json
{
  "recordings":102,
  "events":342,
  "subjects":24
}
```

---

## Analytics Dashboard

GET

```http
/api/v1/dashboard/analytics
```

---

## Trend Dashboard

GET

```http
/api/v1/dashboard/trends
```

---

# 14. Evaluation APIs

---

## Model Metrics

GET

```http
/api/v1/evaluation/models
```

Response

```json
{
  "cnn_accuracy":0.94,
  "transformer_accuracy":0.89
}
```

---

## Retrieval Metrics

GET

```http
/api/v1/evaluation/retrieval
```

---

# 15. Internal AI Service Flow

```text
Recording Uploaded
       │
       ▼
Audio Processing API
       │
       ▼
Detection API
       │
       ▼
Event Storage
       │
       ▼
Temporal API
       │
       ▼
Retrieval API
       │
       ▼
Insight API
```

---

# 16. Future APIs

Future versions may include:

```text
Patient APIs

Authentication APIs

Hospital Integration APIs

Wearable Device APIs

Realtime Streaming APIs
```

---

# API Summary

PRISM exposes six major services:

```text
Dataset Service

Recording Service

Detection Service

Temporal Service

Retrieval Service

Insight Service
```

Together these APIs provide the communication backbone connecting:

```text
Frontend
     │
     ▼
FastAPI
     │
     ▼
AI Models
     │
     ▼
Database
     │
     ▼
RATM Engine
```

while remaining modular, scalable, and compatible with future healthcare deployments.
