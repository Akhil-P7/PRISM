# Database Design Document (V2)

# PRISM

## Pediatric Respiratory Intelligence System

Version: 2.0

Document Type: Database Design Specification

Status: Official Architecture Version

---

# 1. Purpose

This document defines the complete database architecture for PRISM.

The database is designed to support:

* Multi-dataset integration
* Audio processing pipelines
* Cough event detection
* Temporal respiratory intelligence
* Environmental correlation analysis
* Retrieval-Augmented Temporal Modeling (RATM)
* Future healthcare deployment

The architecture prioritizes:

* Scalability
* Explainability
* Dataset traceability
* Retrieval efficiency
* Future extensibility

---

# 2. Database Philosophy

PRISM is not simply a cough detection system.

It is a respiratory intelligence platform.

Therefore the database must support the entire intelligence pipeline.

Instead of storing only audio files and predictions, the database stores the complete evolution of information.

```text
Raw Dataset
      ↓
Recording
      ↓
Audio Segments
      ↓
Detected Events
      ↓
Temporal Features
      ↓
Predictions
      ↓
Insights
      ↓
Memory Objects
```

Every stage remains traceable.

---

# 3. Architectural Principles

The database follows five principles.

## Dataset Agnostic

The schema must support:

* COUGHVID
* Coswara
* ICBHI
* Future hospital datasets
* Future wearable devices

without redesign.

---

## Traceability

Every prediction should be traceable back to:

* Source dataset
* Recording
* Audio segment
* Detection event

---

## Explainability

The database must store enough information to explain why a model generated a prediction.

---

## Retrieval Readiness

The architecture must support similarity search and memory retrieval.

---

## Longitudinal Analysis

The schema must support temporal analysis across multiple recordings and subjects.

---

# 4. Conceptual Data Model

The core hierarchy of PRISM is:

```text
Dataset
   │
   ▼
Subject
   │
   ▼
Recording
   │
   ▼
Segment
   │
   ▼
Event
   │
   ▼
Temporal Features
   │
   ▼
Predictions
   │
   ▼
Insights
   │
   ▼
Memory Store
```

This hierarchy represents the complete respiratory intelligence lifecycle.

---

# 5. Storage Architecture

PRISM uses a hybrid storage architecture.

## Relational Layer

Technology:

* SQLite (Development)
* PostgreSQL (Production)

Purpose:

Store structured information.

---

## Vector Layer

Technology:

* FAISS

Purpose:

Store embeddings and retrieval memory.

---

## File Storage Layer

Purpose:

Store audio files.

Examples:

```text
audio/
    coughvid/
    coswara/
    icbhi/
```

The database stores references, not raw files.

---

# 6. Core Entities

---

# Entity 1: Dataset

Represents the source dataset.

Examples:

* COUGHVID
* Coswara
* ICBHI

---

Fields

| Field       | Type      |
| ----------- | --------- |
| dataset_id  | UUID      |
| name        | String    |
| version     | String    |
| source_url  | String    |
| description | Text      |
| created_at  | Timestamp |

---

Purpose

Maintain dataset provenance.

---

# Entity 2: Subject

Represents an individual participant.

A subject may originate from:

* COUGHVID
* Coswara
* ICBHI
* Future deployment users

---

Fields

| Field             | Type      |
| ----------------- | --------- |
| subject_id        | UUID      |
| dataset_id        | UUID      |
| source_subject_id | String    |
| age               | Integer   |
| age_group         | String    |
| gender            | String    |
| country           | String    |
| region            | String    |
| health_status     | String    |
| created_at        | Timestamp |

---

Purpose

Provide a unified representation of participants across datasets.

---

# Entity 3: Recording

Represents a single audio recording.

---

Fields

| Field               | Type      |
| ------------------- | --------- |
| recording_id        | UUID      |
| subject_id          | UUID      |
| dataset_id          | UUID      |
| file_path           | String    |
| duration            | Float     |
| sample_rate         | Integer   |
| channels            | Integer   |
| recording_type      | String    |
| recording_timestamp | Timestamp |

---

Examples

```text
Cough

Breathing

Speech

Vowel

Counting
```

---

# Entity 4: Segment

Represents a portion of an audio recording.

---

Fields

| Field        | Type  |
| ------------ | ----- |
| segment_id   | UUID  |
| recording_id | UUID  |
| start_time   | Float |
| end_time     | Float |
| duration     | Float |
| confidence   | Float |

---

Purpose

Allow event detection on manageable audio windows.

---

# Entity 5: Event

Represents a detected respiratory event.

Version 1 focuses on cough detection.

---

Fields

| Field      | Type      |
| ---------- | --------- |
| event_id   | UUID      |
| segment_id | UUID      |
| event_type | String    |
| timestamp  | Timestamp |
| duration   | Float     |
| intensity  | Float     |
| confidence | Float     |

---

Supported Events

Version 1

```text
Cough
```

Future

```text
Wheeze

Crackle

Breathing Anomaly
```

---

# 7. Feature Store Layer

A dedicated feature store supports temporal intelligence.

---

# Entity 6: Temporal Features

Generated by the Temporal Transformer pipeline.

---

Fields

| Field              | Type      |
| ------------------ | --------- |
| feature_id         | UUID      |
| subject_id         | UUID      |
| recording_id       | UUID      |
| cough_count        | Integer   |
| avg_duration       | Float     |
| avg_intensity      | Float     |
| night_ratio        | Float     |
| peak_hour          | Integer   |
| cough_burden_score | Float     |
| generated_at       | Timestamp |

---

Purpose

Store aggregated respiratory behavior.

---

Example

```text
Total Coughs: 32

Average Duration: 1.4 sec

Night Ratio: 0.68
```

---

# 8. Environmental Intelligence Layer

Stores environmental context.

---

# Entity 7: Environmental Data

---

Fields

| Field             | Type      |
| ----------------- | --------- |
| environment_id    | UUID      |
| recording_id      | UUID      |
| AQI               | Integer   |
| temperature       | Float     |
| humidity          | Float     |
| weather_condition | String    |
| timestamp         | Timestamp |

---

Future Fields

```text
PM2.5

PM10

Pollen

Dust Index
```

---

Purpose

Enable environmental correlation analysis.

---

# 9. Prediction Layer

Stores outputs from AI models.

---

# Entity 8: Predictions

---

Fields

| Field         | Type      |
| ------------- | --------- |
| prediction_id | UUID      |
| subject_id    | UUID      |
| recording_id  | UUID      |
| trend_class   | String    |
| risk_score    | Float     |
| confidence    | Float     |
| generated_at  | Timestamp |

---

Trend Classes

```text
Stable

Improving

Increasing

Abnormal
```

---

Purpose

Persist model outputs.

---

# 10. Insight Layer

Stores explainable AI outputs.

---

# Entity 9: Insights

---

Fields

| Field         | Type      |
| ------------- | --------- |
| insight_id    | UUID      |
| prediction_id | UUID      |
| insight_text  | Text      |
| generated_at  | Timestamp |

---

Example

```text
Nighttime cough burden increased by 22%
compared to previous observations.
```

---

Purpose

Maintain explainability records.

---

# 11. Retrieval-Augmented Temporal Modeling Layer

This is the core research contribution of PRISM.

---

# Entity 10: Memory Objects

Represents retrievable respiratory experiences.

---

Fields

| Field        | Type      |
| ------------ | --------- |
| memory_id    | UUID      |
| subject_id   | UUID      |
| recording_id | UUID      |
| summary      | Text      |
| embedding_id | String    |
| created_at   | Timestamp |

---

Example Summary

```text
High AQI

Elevated nighttime coughing

Moderate respiratory risk
```

---

Purpose

Support retrieval of similar respiratory situations.

---

# 12. Vector Database Design

The vector database stores semantic representations.

---

Embedding Sources

```text
Temporal Features

Environmental Features

Predictions

Historical Trends
```

---

Embedding Dimension

Version 1

```text
512 Dimensions
```

---

Similarity Metric

```text
Cosine Similarity
```

---

Purpose

Retrieve:

* Similar respiratory episodes
* Similar environmental patterns
* Similar cough progression trends

---

# 13. Entity Relationships

Dataset

```text
1 → Many Subjects
```

---

Subject

```text
1 → Many Recordings
```

---

Recording

```text
1 → Many Segments

1 → Many Environmental Records

1 → Many Feature Records
```

---

Segment

```text
1 → Many Events
```

---

Prediction

```text
1 → Many Insights
```

---

Memory Object

```text
1 → 1 Vector Embedding
```

---

# 14. Data Lifecycle

Stage 1

Dataset Imported

↓

Dataset Record Created

---

Stage 2

Subject Created

↓

Recording Created

---

Stage 3

Audio Segmentation

↓

Segment Records Created

---

Stage 4

CNN Detection

↓

Event Records Created

---

Stage 5

Feature Engineering

↓

Temporal Features Stored

---

Stage 6

Transformer Analysis

↓

Predictions Generated

---

Stage 7

Insight Generation

↓

Insights Stored

---

Stage 8

Memory Construction

↓

Embeddings Stored

---

# 15. Indexing Strategy

High-priority indexes:

---

Datasets

```sql
dataset_id
```

---

Subjects

```sql
subject_id
dataset_id
```

---

Recordings

```sql
recording_id
subject_id
```

---

Events

```sql
event_id
timestamp
```

---

Temporal Features

```sql
subject_id
recording_id
```

---

Predictions

```sql
subject_id
risk_score
```

---

Memory Objects

```sql
memory_id
```

---

# 16. Future Healthcare Extensions

Future versions may introduce:

## Patient Entity

For real-world deployment.

---

## Medication Records

```text
Medication

Dosage

Schedule
```

---

## Clinical Diagnosis

```text
Asthma

COPD

Pneumonia
```

---

## Hospital Integration

```text
Electronic Health Records
```

---

## Wearable Device Support

```text
Microphone Streams

Sensor Data

Realtime Monitoring
```

---

# 17. Database Summary

PRISM adopts a layered database architecture.

```text
Dataset Layer
        │
        ▼
Subject Layer
        │
        ▼
Recording Layer
        │
        ▼
Event Layer
        │
        ▼
Feature Store Layer
        │
        ▼
Prediction Layer
        │
        ▼
Insight Layer
        │
        ▼
Memory Layer
```

This architecture supports dataset integration, temporal intelligence, environmental analysis, explainable AI, and Retrieval-Augmented Temporal Modeling while remaining scalable enough for future healthcare deployment.