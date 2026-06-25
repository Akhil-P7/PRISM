# PRISM: Unified Data Architecture & Ingestion Pipeline

This document serves as a presentation of the **PRISM Data Foundation Layer**. It explains how we wrangle chaotic, diverse healthcare datasets into a clean, unified format, and how this foundation powers the rest of the PRISM AI ecosystem.

---

## 1. The Challenge: Dataset Chaos

In respiratory audio analysis, no two datasets are the same. We are currently utilizing three major open-source datasets, each with entirely different structures:

*   **COUGHVID:** ~34,000 crowdsourced recordings. Metadata is in a single `metadata_compiled.csv` file. It uses columns like `uuid`, `fever_muscle_pain`, and `status`.
*   **Coswara:** ~1,200 clinical recordings. Metadata is in `combined_data.csv`, using completely different, highly abbreviated column names like `a` for age, `g` for gender, and `covid_status`. Audio is categorized into folders like `cough-shallow` and `cough-heavy`.
*   **ICBHI:** ~900 clinical stethoscope recordings. **It has no CSV metadata file.** All metadata (patient ID, body location, equipment) is bizarrely encoded directly into the filenames (e.g., `101_1b1_Al_sc_Meditron.wav`), and timestamps for crackles/wheezes are stored in matching `.txt` files.

**The Problem:** If an ML Engineer wants to train a model to "detect coughs in males over 40", they would normally have to write three completely different sets of code to query the three different datasets.

---

## 2. The Solution: The Unified Data Model

We solved this chaos by creating the **Unified Data Model** in PostgreSQL. Instead of storing the datasets in their native formats, we force them to map into three core, standardized tables.

### Core Tables

#### 1. `datasets`
Tracks the source of the data.
*   `id`, `name`, `version`, `description`

#### 2. `subjects` (Patients)
The human being who provided the audio. All dataset-specific demographics are normalized to this standard.
*   `id` (Primary Key)
*   `dataset_id` (Foreign Key)
*   `source_subject_id` (The original ID, e.g., Coswara's 'vK2bLRNz...')
*   `age` (Integer)
*   `gender` (String: 'Male', 'Female', etc.)
*   `respiratory_condition` (String: 'Healthy', 'COVID-19', 'Symptomatic')
*   `has_fever` (Boolean)
*   `is_smoker` (Boolean)

#### 3. `recordings` (Audio)
The actual audio files linked to a subject.
*   `id` (Primary Key)
*   `subject_id` (Foreign Key)
*   `file_path` (Where the audio lives on disk)
*   `duration` (Float in seconds)
*   `equipment` (e.g., 'Smartphone', 'WelchAllyn Meditron')
*   `is_cough` (Boolean)

---

### Detailed Data Model Specification

The unified PostgreSQL schema consists of three core tables, each backed by a SQLAlchemy model in `database/models`.

| Table | SQLAlchemy Model | Primary Key | Foreign Keys | Core Fields | Description |
|-------|------------------|-------------|--------------|------------|-------------|
| `datasets` | `Dataset` | `id` (UUID) | — | `name`, `version`, `description` | Captures provenance of each source dataset (COUGHVID, Coswara, ICBHI). |
| `subjects` | `Subject` | `id` (UUID) | `dataset_id` → `datasets.id` | `source_subject_id`, `age`, `gender`, `respiratory_condition`, `has_fever`, `is_smoker` | One row per participant; demographics are normalized across datasets. |
| `recordings` | `Recording` | `id` (UUID) | `subject_id` → `subjects.id` | `file_path`, `duration`, `equipment`, `is_cough` | Points to an audio file (stored on disk) and its basic acoustic metadata. |

**Field details**:
- `source_subject_id` retains the original identifier from the source (e.g., Coswara's `vK2bLRNz...`). It is **not** unique globally because different datasets may reuse IDs; uniqueness is enforced together with `dataset_id`.
- `age` is stored as an integer (years) and may be `NULL` when the source does not provide it (ICBHI).
- `gender` is a short string; values are normalised to `Male`, `Female`, or `Other`.
- `respiratory_condition` captures the clinical label (e.g., `Healthy`, `COVID-19`, `Symptomatic`).
- `has_fever` and `is_smoker` are booleans, defaulting to `NULL` when unavailable.
- `duration` records the length of the audio clip in seconds; it is optional because some source metadata omit it.
- `equipment` records the capture device (e.g., `Smartphone`, `WelchAllyn Meditron`).
- `is_cough` is a boolean flag indicating whether the clip contains a cough sound; for non‑cough datasets (ICBHI) it is set to `FALSE`.

### Model ↔ Table Mapping
- **`database/models/dataset.py`** defines class `Dataset` with columns matching the `datasets` table.
- **`database/models/subject.py`** defines class `Subject` with a many‑to‑one relationship to `Dataset` via `dataset_id`.
- **`database/models/recording.py`** defines class `Recording` with a many‑to‑one relationship to `Subject` via `subject_id`.

These models are used by the ingestion adapters (`CoughvidAdapter`, `CoswaraAdapter`, `IcbhiAdapter`) through the `BaseAdapter.ingest` method, which creates instances of the models and persists them with SQLAlchemy sessions.

---

## 3. The Ingestion Pipeline

To move data from the ZIP files into our Unified PostgreSQL database, we built an object-oriented ingestion pipeline using the **Adapter Pattern**.

```mermaid
graph TD
    subgraph Raw Data (ZIP Archives)
        A[COUGHVID.zip<br>metadata_compiled.csv]
        B[Coswara.zip<br>combined_data.csv]
        C[ICBHI.zip<br>Filenames & .txt]
    end

    subgraph PRISM Adapters
        D[CoughvidAdapter]
        E[CoswaraAdapter]
        F[IcbhiAdapter]
    end

    subgraph Unified Storage
        G[(PostgreSQL<br>Unified Schema)]
    end

    A --> D
    B --> E
    C --> F

    D -->|Normalizes to Subject/Recording| G
    E -->|Normalizes to Subject/Recording| G
    F -->|Parses Regex to Subject/Recording| G
```

### How the Adapters Work
1.  **BaseAdapter:** Defines the master contract. Every adapter *must* implement an `extract_metadata()` method that returns standard Python dictionaries matching our Unified Schema. It also handles the complex PostgreSQL insertion and duplicate-checking logic safely using SQLAlchemy.
2.  **Dataset-Specific Adapters:** Classes like `CoswaraAdapter` inherit from `BaseAdapter`. Their only job is to open the ZIP, read the messy data, translate `a` to `age` and `g` to `gender`, and hand it back to the BaseAdapter.

**Crucially:** We do not unzip the massive audio files during metadata ingestion. The adapters read the CSV/TXT files *directly out of the compressed ZIP archives* in memory, keeping the pipeline lightning fast (extracting 35,000+ subjects in under 16 minutes).

---

## 4. Where is this data used later?

This Unified Data Foundation is the bedrock of the entire PRISM project.

### Phase 2: Machine Learning Pipeline
When we build our Convolutional Neural Networks (CNNs) to detect respiratory diseases, the PyTorch `Dataset` loaders will directly query this PostgreSQL database.
Because the schema is unified, the PyTorch code is incredibly simple:
```sql
-- The ML Pipeline only needs to run this one query to get training data from ALL datasets simultaneously
SELECT r.file_path, s.respiratory_condition
FROM recordings r
JOIN subjects s ON r.subject_id = s.id
WHERE r.is_cough = true AND s.gender = 'Male'
```
The ML model never knows (or cares) if the audio came from Coswara or COUGHVID.

### Phase 3: The Dashboard API
The Streamlit frontend and FastAPI backend will use this unified database to display analytics. For example, generating a pie chart of "Age Distribution across all Datasets" is now a single, instantaneous SQL query, rather than a complex Pandas script joining multiple messy CSVs.
