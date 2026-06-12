# Data Ingestion & Unified Data Model

# PRISM

## Pediatric Respiratory Intelligence System

Version: 1.0

Document Type: Data Engineering Specification

---

# 1. Purpose

This document defines:

* Dataset selection
* Dataset integration strategy
* Unified schema design
* Data normalization rules
* Data ingestion pipeline
* Mapping between external datasets and PRISM

This document serves as the bridge between:

```text
Public Datasets
        ↓
AI Models
        ↓
Database Layer
```

---

# 2. Selected Datasets

After evaluation, PRISM will use:

## Primary Dataset

### COUGHVID V3

Role:

Primary cough detection dataset.

Reason:

* Largest cough dataset
* Real-world recording conditions
* Rich metadata
* Diverse cough characteristics

Used For:

* CNN cough detector training
* Feature extraction
* Generalization testing

---

## Secondary Dataset

### Coswara

Role:

Supplementary cough dataset.

Reason:

* High quality recordings
* Controlled recording environment
* Rich participant metadata

Used For:

* Validation
* Robustness testing
* Metadata analysis

---

## Auxiliary Dataset

### ICBHI 2017

Role:

Respiratory pathology dataset.

Reason:

* Clinical respiratory sounds
* Pediatric representation
* Disease annotations

Used For:

* Generalization testing
* Respiratory pattern analysis
* Future respiratory event expansion

---

# 3. Data Modeling Philosophy

The datasets have different structures.

Instead of forcing them into one dataset format, PRISM introduces a common abstraction.

---

## Core Entity Hierarchy

```text
Dataset
    ↓
Subject
    ↓
Recording
    ↓
Segment
    ↓
Event
```

This hierarchy works for:

* COUGHVID
* Coswara
* ICBHI
* Future hospital datasets
* Future wearable devices

---

# 4. Unified Data Model

---

## Dataset Entity

Represents source dataset.

Fields:

| Field        | Type   |
| ------------ | ------ |
| dataset_id   | UUID   |
| dataset_name | String |
| version      | String |
| source_url   | String |

Examples:

```text
COUGHVID

Coswara

ICBHI
```

---

## Subject Entity

Represents a person.

Fields:

| Field             | Type    |
| ----------------- | ------- |
| subject_id        | UUID    |
| dataset_id        | UUID    |
| source_subject_id | String  |
| age               | Integer |
| age_group         | String  |
| gender            | String  |
| country           | String  |
| region            | String  |

---

Age Groups

```text
Child

Adolescent

Adult

Senior
```

---

## Recording Entity

Represents a single recording.

Fields:

| Field               | Type      |
| ------------------- | --------- |
| recording_id        | UUID      |
| subject_id          | UUID      |
| dataset_id          | UUID      |
| audio_path          | String    |
| sample_rate         | Integer   |
| duration            | Float     |
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

## Segment Entity

Represents audio chunks.

Fields:

| Field        | Type  |
| ------------ | ----- |
| segment_id   | UUID  |
| recording_id | UUID  |
| start_time   | Float |
| end_time     | Float |
| confidence   | Float |

---

Example

```text
Recording

0s -------------------- 10s

Segment

2.5s ---- 3.8s
```

---

## Event Entity

Represents detected respiratory events.

Fields:

| Field      | Type   |
| ---------- | ------ |
| event_id   | UUID   |
| segment_id | UUID   |
| event_type | String |
| duration   | Float  |
| intensity  | Float  |
| confidence | Float  |

---

Event Types

Version 1:

```text
Cough
```

Future:

```text
Cough

Wheeze

Crackle

Breathing Anomaly
```

---

# 5. Dataset Field Mapping

---

## COUGHVID Mapping

| COUGHVID              | PRISM         |
| --------------------- | ------------- |
| uuid                  | recording_id  |
| age                   | age           |
| gender                | gender        |
| country               | country       |
| respiratory_condition | health_status |
| audio_file            | audio_path    |

---

## Coswara Mapping

| Coswara         | PRISM             |
| --------------- | ----------------- |
| subject_id      | source_subject_id |
| age             | age               |
| gender          | gender            |
| country         | country           |
| state           | region            |
| covid_status    | health_status     |
| cough_heavy.wav | recording         |

---

## ICBHI Mapping

| ICBHI      | PRISM             |
| ---------- | ----------------- |
| patient_id | source_subject_id |
| age        | age               |
| gender     | gender            |
| diagnosis  | health_status     |
| wav_file   | audio_path        |

---

# 6. Data Normalization Rules

Different datasets store information differently.

---

## Gender Normalization

Input:

```text
M

Male

male
```

Output:

```text
Male
```

---

Input:

```text
F

Female

female
```

Output:

```text
Female
```

---

## Age Normalization

Input:

```text
Age = 12
```

Output:

```text
Child
```

---

Input:

```text
Age = 45
```

Output:

```text
Adult
```

---

# Health Status Normalization

COUGHVID:

```text
COVID

Asthma

Healthy
```

Coswara:

```text
Positive

Negative

Recovered
```

ICBHI:

```text
COPD

Pneumonia

Healthy
```

---

Unified Labels

```text
Healthy

Respiratory Disease

Unknown
```

---

# 7. Data Ingestion Pipeline

---

## Stage 1

Raw Dataset Import

```text
Dataset
      ↓
Adapter
```

---

## Stage 2

Validation

Checks:

* Missing files
* Corrupt audio
* Invalid metadata

---

## Stage 3

Normalization

Standardize:

* Labels
* Age groups
* Gender values
* Health categories

---

## Stage 4

Audio Standardization

Convert all recordings to:

```text
WAV

44.1 kHz

Mono
```

---

## Stage 5

Feature Generation

Generate:

```text
Mel Spectrograms

Metadata Features
```

---

## Stage 6

Storage

Store:

```text
Unified Dataset

Database

Feature Cache
```

---

# 8. Adapter Architecture

Each dataset receives a dedicated adapter.

---

## COUGHVID Adapter

```text
COUGHVID
      ↓
COUGHVID Adapter
      ↓
Unified Schema
```

---

## Coswara Adapter

```text
Coswara
      ↓
Coswara Adapter
      ↓
Unified Schema
```

---

## ICBHI Adapter

```text
ICBHI
      ↓
ICBHI Adapter
      ↓
Unified Schema
```

---

# 9. Directory Structure

```text
datasets/

    raw/

        coughvid/

        coswara/

        icbhi/

    processed/

        unified/

    metadata/

        unified_records.csv
```

---

# 10. Integration with PRISM Database

The ingestion pipeline feeds directly into:

```text
Dataset
      ↓
Subject
      ↓
Recording
      ↓
Segment
      ↓
Event
```

The Event table becomes the source for:

```text
Temporal Transformer
```

while Subject + Recording metadata become the source for:

```text
Retrieval-Augmented Temporal Modeling
```

---

# 11. Future Expansion

The model is intentionally extensible.

Future datasets may include:

* Pediatric hospital recordings
* Wearable microphone streams
* Home monitoring systems
* Environmental sensors

No schema redesign should be required.

---

# Final Unified Flow

```text
COUGHVID
        │
Coswara │
        │
ICBHI   │
        ▼
Dataset Adapters
        ▼
Normalization Layer
        ▼
Unified Data Model
        ▼
Database
        ▼
Feature Generation
        ▼
CNN Detector
        ▼
Temporal Transformer
        ▼
RATM Engine
```

This data model ensures that PRISM can integrate heterogeneous respiratory datasets while maintaining a consistent structure for training, evaluation, retrieval, and future deployment.
