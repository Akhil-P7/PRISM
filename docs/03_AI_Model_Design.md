# AI Model Design Document

# PRISM

## Patient Respiratory Intelligence System

**Version:** 1.0

**Document Type:** AI Architecture & Model Design Specification

---

# 1. Purpose

This document defines the complete Artificial Intelligence architecture used in PRISM.

It explains:

* Model selection rationale
* Data representations
* Learning objectives
* Model interactions
* Retrieval-Augmented Temporal Modeling (RATM)

The goal is to build a clinically meaningful respiratory intelligence system rather than a simple cough classifier.

---

# 2. AI Philosophy

PRISM treats respiratory monitoring as a multi-stage intelligence problem.

Instead of:

Audio → Prediction

PRISM follows:

Audio → Event Detection → Temporal Understanding → Disease Classification → Retrieval → Clinical Insight

Each stage solves a distinct problem, contributing to a highly explainable final assessment.

---

# 3. AI Pipeline Overview

```text
Raw Audio
      │
      ▼
Audio Representation Layer (Mel Spectrograms)
      │
      ▼
Cough Detection Layer (ResNet-18)
      │
      ▼
Event Generation Layer
      │
      ├───────────────────────┐
      ▼                       ▼
Temporal Intelligence   Disease Classification
      │                       │
      └───────────┬───────────┘
                  ▼
          Retrieval Layer (TurboVec)
                  │
                  ▼
      Clinical Insight Generation
```

---

# 4. Audio Representation Layer

## Objective

Transform raw audio into machine-learnable representations.

---

# Why Not Feed Raw Audio Directly?

Raw audio contains:

* Background noise
* Device-specific artifacts
* Large dimensionality

Training directly on waveform data requires massive datasets and large compute resources, neither of which is practical for our current deployment strategy.

---

# Chosen Representation

## Mel Spectrogram

The Mel Spectrogram converts audio into a time-frequency image.

Advantages:

* Preserves temporal information
* Preserves frequency information
* Human-auditory inspired
* Well-established in audio AI research
* Compatible with standard image-based CNN architectures

---

# Why Mel Spectrogram Instead of MFCC?

Many legacy audio projects use:

MFCC → LSTM

However, MFCCs compress information aggressively and discard spectral details, making deep feature learning and transfer learning difficult. Mel Spectrograms provide a richer representation for modern CNNs.

---

# 5. Cough Detection Layer

## Objective

Determine whether a cough event exists within an audio segment and extract a rich acoustic embedding.

---

# Proposed Architecture

**Model:** ResNet-18 CNN

```text
Mel Spectrogram
        │
        ▼
   ResNet Blocks
        │
        ▼
Global Average Pooling
        │
        ▼
   512-D Embedding
        │
        ▼
Fully Connected Layer
        │
        ▼
     Sigmoid
```

Output:

1. `P(Cough)` - Probability that the segment contains a cough.
2. `512-D Acoustic Embedding` - A dense vector representing the acoustic signature of the cough, used later for disease classification and vector retrieval.

---

# 6. Event Generation Layer

Purpose:

Convert raw frame-level predictions into structured respiratory events.

---

Input:

Audio Segment + P(Cough)

Output:

```json
{
  "timestamp": "2026-01-10T21:04:02",
  "confidence": 0.96,
  "duration": 1.2,
  "intensity": 0.84
}
```

This separation ensures temporal models learn respiratory behavior (frequency, clustering) rather than raw acoustic noise.

---

# 7. Temporal Intelligence Layer

## Objective

Understand how coughing evolves over a 30-day window to predict clinical trajectories.

---

# Why Transformer?

Traditional RNNs/LSTMs suffer from sequential bottlenecks and weak long-range reasoning. Transformers use attention mechanisms to model 30-day histories in parallel, offering stronger temporal reasoning and better interpretability.

---

# Input Features

For each time window (daily aggregates):

* Cough Count
* Average Duration
* Average Intensity
* Night Ratio
* Inter-Cough Interval

---

# Output Trajectories

The Temporal Transformer predicts one of four distinct trends:

* `Stable Trend`
* `Improving Trend`
* `Increasing Trend`
* `Abnormal Spike`

---

# 8. Disease Classification Layer

Purpose:

Predict specific respiratory conditions based purely on acoustic signatures.

---

# Architecture

A Multi-Layer Perceptron (MLP) head that takes the `512-D Acoustic Embedding` from the Cough Detector as input.

Output:

Probability distribution over specific diseases (e.g., Asthma, Bronchitis, Pneumonia, Pertussis).

---

# 9. Retrieval-Augmented Temporal Modeling (RATM)

This is the core research contribution of PRISM.

---

# Problem

Traditional models predict (e.g., "Asthma: 80%"). They do not explain *why*. Most AI lacks historical context to justify its assessments.

---

# Proposed Solution

Retrieval-Augmented Temporal Modeling (RATM). By retrieving historically similar patient cases, the AI can ground its predictions in real-world evidence.

---

## Components

### Vector Database (TurboVec)
Stores historical 512-D acoustic embeddings mapped to known patient diagnoses and trajectories.

### Retriever
Searches TurboVec for the top-$K$ embeddings mathematically closest to the current patient's embedding.

### Memory Builder
Assembles the current patient's temporal trajectory, demographic data, and the retrieved historical cases into a structured "Clinical Memory."

### Insight Generator
Uses rule-based templating to synthesize the Clinical Memory into a human-readable Clinical Insight Report.

---

# 10. Embedding Strategy

Purpose:

Represent respiratory states as dense vectors.

Output:

`512-Dimensional Vector` (Extracted from the ResNet-18 Cough Detector).

These vectors are stored inside the vector database and queried using Cosine Similarity.

---

# 11. Vector Database Design

Candidate Systems:

### TurboVec (Selected)

Advantages:
* Built on Google Research's TurboQuant algorithm.
* No training step — data-oblivious quantization.
* 16x memory compression (4-bit) vs float32.
* 10-19% faster search than FAISS on ARM.
* Ultra-lightweight and highly compatible with edge deployments.

### ChromaDB
Advantages: Metadata support and easy API. (Rejected for V1 due to unnecessary overhead compared to TurboVec's raw speed).

---

# 12. Clinical Insight Generator

Purpose:

Convert the retrieved context into an actionable, human-readable report for clinicians.

Input:
* Current Temporal Trajectory
* Predicted Disease
* Top-K Retrieved Historical Cases

Output:
A structured text report detailing:
* Narrative Summary
* Severity Assessment (Low/Medium/High)
* Specific Clinical Observations (e.g., "Nighttime coughing is highly prominent, typical of asthma presentations.")

---

# 13. Model Evaluation Strategy

## Cough Detection
Metrics: Accuracy, Precision, Recall, F1 Score, ROC AUC.

## Temporal Intelligence
Metrics: Trend Classification Accuracy, Sequence Prediction Accuracy.

## Disease Classification
Metrics: Multi-class F1 Score, Confusion Matrix.

## Retrieval Layer
Metrics: Retrieval Precision@K, Retrieval Recall@K.

---

# 14. Model Training Strategy

Phase 1: Train ResNet-18 Cough Detector & Extract Embeddings.

Phase 2: Train Disease Classifier on Acoustic Embeddings.

Phase 3: Generate Temporal Event Datasets.

Phase 4: Train Temporal Transformer.

Phase 5: Build TurboVec Retrieval Database with Historical Embeddings.

Phase 6: Implement RATM Pipeline (Memory Builder + Insight Generator).

---

# 15. Future Model Evolution

Future versions may include:

## Environmental Correlation Engine
Integrating AQI, humidity, and temperature data as additional features for the Temporal Transformer.

## Audio Foundation Models
Upgrading from ResNet-18 to large pre-trained audio models (e.g., Wav2Vec2, Audio Spectrogram Transformer, BEATs).

## Multi-Modal Learning
Fusing acoustic embeddings, temporal sequences, and environmental time-series into a single joint-embedding space.

---

# 16. AI Architecture Summary

PRISM uses a hierarchical intelligence architecture:

```text
Audio
   │
   ▼
Mel Spectrogram
   │
   ▼
ResNet-18 Extractor
   │
   ├───────────┐
   ▼           ▼
Events     Embeddings
   │           │
   ▼           ▼
Temporal    Disease Classifier &
Transformer   TurboVec Retrieval
   │           │
   └─────┬─────┘
         ▼
    RATM Engine
         │
         ▼
Clinical Insight Generation
```

This architecture balances practicality, research depth, computational feasibility, and clinical explainability while remaining highly scalable.ation
```

This architecture balances practicality, research depth, computational feasibility, explainability, and future scalability while remaining achievable within a student-led research project.
