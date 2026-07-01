# Data Ingestion & Unified Data Model

# PRISM

## Patient Respiratory Intelligence System

**Version:** 1.0
**Document Type:** Data Engineering Specification

---

# 1. Purpose

This document defines:
* Dataset selection
* Dataset integration strategy
* Unified schema design
* Data normalization rules
* Data ingestion pipeline
* Mapping between external datasets and PRISM

This document serves as the bridge between Public Datasets, AI Models, and the Database Layer.

---

# 2. Selected Datasets

After evaluation, PRISM incorporates the following datasets:

## Primary Dataset: COUGHVID V3
**Role:** Primary cough detection dataset.
**Reason:** Largest cough dataset, real-world recording conditions, rich metadata.
**Used For:** CNN cough detector training, feature extraction, generalization testing.

## Secondary Dataset: Coswara
**Role:** Supplementary cough dataset.
**Reason:** High quality recordings, controlled recording environment, rich participant metadata.
**Used For:** Validation, robustness testing, metadata analysis.

## Auxiliary Dataset: ICBHI 2017
**Role:** Respiratory pathology dataset.
**Reason:** Clinical respiratory sounds, Patient representation, disease annotations.
**Used For:** Generalization testing, respiratory pattern analysis, future expansion.

---

# 3. Data Modeling Philosophy

The datasets have different structures. Instead of forcing them into one strict external format, PRISM introduces a common internal abstraction hierarchy:

`Dataset $\rightarrow$ Subject $\rightarrow$ Recording $\rightarrow$ Segment $\rightarrow$ Event`

---

# 4. Unified Data Model

## Entity: Dataset
Represents the source dataset.
* **Fields:** `id` (UUID), `name` (String), `version` (String), `description` (String)

## Entity: Subject
Represents a person.
* **Fields:** `id` (UUID), `dataset_id` (UUID), `source_subject_id` (String), `age` (Integer), `gender` (String), `respiratory_condition` (String), `has_fever` (Boolean), `is_smoker` (Boolean)

## Entity: Recording
Represents a single recording.
* **Fields:** `id` (UUID), `subject_id` (UUID), `file_path` (String), `duration` (Float), `equipment` (String), `is_cough` (Boolean)

## Entities: Segment & Event (Conceptual / V2 Roadmap)
Used during inference and RATM processing, but persisted as features rather than raw SQL rows in V1.
* **Segment:** Represents audio chunks (`start_time`, `end_time`, `confidence`).
* **Event:** Represents detected respiratory events (`event_type`, `duration`, `intensity`, `confidence`). Version 1 focuses exclusively on **Cough**.

---

# 5. Dataset Field Mapping

Different datasets are normalized into PRISM's unified schema.

## COUGHVID Mapping
| COUGHVID Field        | PRISM Field           |
| --------------------- | --------------------- |
| `uuid`                | `recording.file_path` / `subject.source_subject_id` |
| `age`                 | `subject.age`         |
| `gender`              | `subject.gender`      |
| `respiratory_condition`| `subject.respiratory_condition` |

## Coswara Mapping
| Coswara Field         | PRISM Field           |
| --------------------- | --------------------- |
| `subject_id`          | `subject.source_subject_id` |
| `age`                 | `subject.age`         |
| `gender`              | `subject.gender`      |
| `covid_status`        | `subject.respiratory_condition` |
| `cough_heavy.wav`     | `recording.file_path` |

## ICBHI Mapping
| ICBHI Field           | PRISM Field           |
| --------------------- | --------------------- |
| `patient_id`          | `subject.source_subject_id` |
| `age`                 | `subject.age`         |
| `gender`              | `subject.gender`      |
| `diagnosis`           | `subject.respiratory_condition` |
| `wav_file`            | `recording.file_path` |

---

# 6. Data Normalization Rules

## Gender Normalization
* "M", "male", "Male" $\rightarrow$ `Male`
* "F", "female", "Female" $\rightarrow$ `Female`

## Health Status Normalization
Mapped into broader diagnostic categories for unified embeddings:
* COUGHVID: COVID, Asthma, Healthy
* Coswara: Positive, Negative, Recovered
* ICBHI: COPD, Pneumonia, Healthy

*Unified Representation:* `Healthy`, `Respiratory Disease`, `Unknown`.

---

# 7. Data Ingestion Pipeline

1. **Raw Dataset Import:** Download and extract COUGHVID, Coswara, ICBHI.
2. **Validation:** Check for missing files, corrupt audio, invalid metadata.
3. **Normalization:** Standardize labels, gender values, and health categories.
4. **Audio Standardization:** Convert all recordings to 16 kHz Mono WAV format (required for ResNet-18 processing).
5. **Feature Generation:** Extract Mel Spectrograms and Metadata Features.
6. **Storage:** Populate SQLite/PostgreSQL Database and generate `embeddings_metadata.csv`.

---

# 8. Directory Structure

```text
datasets/
    raw/
        coughvid/
        coswara/
        icbhi/
    processed/
        unified_audio/
    metadata/
        unified_records.csv
```

---

# 9. Future Expansion

The data ingestion model is intentionally extensible. Future datasets may include:
* Patient hospital recordings
* Wearable microphone streams
* Home monitoring systems

No schema redesign will be required to ingest these new modalities.
