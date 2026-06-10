# AI Model Design Document

# PRISM

## Pediatric Respiratory Intelligence System

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

Audio → Event Detection → Temporal Understanding → Retrieval → Clinical Insight

Each stage solves a distinct problem.

---

# 3. AI Pipeline Overview

```text
Raw Audio
      │
      ▼
Audio Representation Layer
      │
      ▼
Cough Detection Layer
      │
      ▼
Event Generation Layer
      │
      ▼
Temporal Intelligence Layer
      │
      ▼
Environmental Intelligence Layer
      │
      ▼
Retrieval Layer
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

Training directly on waveform data requires:

* Massive datasets
* Large compute resources

Neither is practical for our project.

---

# Chosen Representation

## Mel Spectrogram

The Mel Spectrogram converts audio into a time-frequency image.

Advantages:

* Preserves temporal information
* Preserves frequency information
* Human-auditory inspired
* Well-established in audio AI research

---

# Why Mel Spectrogram Instead of MFCC?

Many student projects use:

MFCC → LSTM

However MFCCs were designed decades ago for speech recognition.

They compress information aggressively.

---

### MFCC Limitations

* Loses spectral detail
* Difficult for deep feature learning
* Limits transfer learning

---

### Mel Spectrogram Benefits

* Richer representation
* Compatible with CNNs
* Better visual pattern learning
* Better transfer learning

---

## Decision

Primary Representation:

Mel Spectrogram

Secondary Baseline:

MFCC

---

# 5. Cough Detection Layer

## Objective

Determine whether a cough event exists within an audio segment.

---

# Candidate Models

### Option 1

CNN

### Option 2

CRNN

### Option 3

Transformer Audio Models

### Option 4

Audio Foundation Models

---

# Model Evaluation

## CNN

Advantages:

* Fast training
* Small dataset friendly
* Excellent local pattern recognition
* Easy deployment

Disadvantages:

* Limited temporal reasoning

---

## CRNN

Advantages:

* CNN + Sequence Modeling

Disadvantages:

* Increased complexity
* Harder optimization

---

## Audio Transformers

Advantages:

* State-of-the-art

Disadvantages:

* Large dataset requirements
* High computational cost

---

# Decision

Version 1:

CNN

Future:

Audio Transformer Benchmark

---

# Proposed CNN Architecture

```text
Mel Spectrogram
        │
        ▼
Conv Block 1
        │
        ▼
Conv Block 2
        │
        ▼
Conv Block 3
        │
        ▼
Global Average Pooling
        │
        ▼
Fully Connected Layer
        │
        ▼
Sigmoid
```

Output:

P(Cough)

---

# 6. Event Generation Layer

Purpose:

Convert predictions into structured respiratory events.

---

Input:

```text
Audio Segment

Prediction = Cough
```

---

Output:

```json
{
  "timestamp": "2026-01-10T21:04:02",
  "confidence": 0.96,
  "duration": 1.2,
  "intensity": 0.84
}
```

---

# Why Event Abstraction?

Temporal models should learn respiratory behavior.

Not raw acoustics.

This separation improves explainability.

---

# 7. Temporal Intelligence Layer

## Objective

Understand how coughing evolves over time.

---

# Why Not Use LSTM?

Traditional architecture:

```text
MFCC
 ↓
LSTM
 ↓
Prediction
```

Problems:

* Sequential bottleneck
* Vanishing gradients
* Weak long-range reasoning
* Limited interpretability

---

# Why Transformer?

Transformers use attention mechanisms.

Benefits:

* Long-range dependency modeling
* Parallel processing
* Better interpretability
* Stronger temporal reasoning

---

# Input Features

For each time window:

```text
Cough Count

Average Duration

Average Intensity

Night Ratio

Inter-Cough Interval
```

---

# Temporal Transformer Architecture

```text
Event Sequence
       │
       ▼
Feature Embedding
       │
       ▼
Positional Encoding
       │
       ▼
Transformer Encoder
       │
       ▼
Temporal Representation
       │
       ▼
Prediction Head
```

---

# Outputs

Examples:

* Stable Trend
* Increasing Trend
* Decreasing Trend
* Abnormal Spike

---

# 8. Environmental Intelligence Layer

Purpose:

Study environmental influence.

---

Input Features

```text
AQI

Temperature

Humidity

Date

Time
```

---

# Modeling Strategy

Initially:

Statistical Correlation

Methods:

* Pearson Correlation
* Spearman Correlation

---

Future:

Environmental Transformer

---

Output

```text
Environmental Risk Score
```

---

# 9. Retrieval-Augmented Temporal Modeling (RATM)

This is the core research contribution.

---

# Problem

Traditional models predict.

They do not explain.

Example:

```text
Prediction:
Risk Increasing
```

Question:

Why?

Most models cannot answer.

---

# Proposed Solution

Retrieval-Augmented Temporal Modeling

---

## Components

### Temporal Transformer

Learns trends.

---

### Memory Store

Stores:

* Historical trends
* Event summaries
* Environmental conditions

---

### Retriever

Searches:

* Similar trends
* Similar respiratory episodes

---

### Clinical Knowledge Base

Contains:

* Medical literature
* Guidelines
* Research summaries

---

### Explanation Layer

Generates explanations.

---

# RATM Architecture

```text
Current Trend
        │
        ▼
Temporal Embedding
        │
        ▼
Retriever
        │
 ┌──────┴──────┐
 ▼             ▼
Historical   Clinical
Memory       Knowledge
 │             │
 └──────┬──────┘
        ▼
Context Builder
        ▼
Explanation Generator
```

---

# 10. Embedding Strategy

Purpose:

Represent respiratory states as vectors.

---

Embedding Inputs

```text
Trend Features

Environmental Features

Historical Features
```

---

Output

```text
512-Dimensional Vector
```

---

These vectors are stored inside the vector database.

---

# 11. Vector Database Design

Candidate Systems:

### FAISS

Advantages:

* Fast
* Lightweight
* Research friendly

### ChromaDB

Advantages:

* Metadata support
* Easy API

---

Decision

Version 1:

FAISS

---

# 12. Explanation Layer

Purpose:

Convert retrieved context into human-readable observations.

---

Input

```text
Current Pattern

Retrieved Cases

Environmental Data
```

---

Output

Example:

"Nighttime cough frequency increased by 24% compared to the previous week and resembles historical periods associated with elevated pollution exposure."

---

# 13. Model Evaluation Strategy

## Cough Detection

Metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC AUC

---

## Temporal Intelligence

Metrics:

* Trend Classification Accuracy
* Sequence Prediction Accuracy

---

## Retrieval Layer

Metrics:

* Retrieval Precision@K
* Retrieval Recall@K

---

## Explanation Layer

Metrics:

* Clinical Relevance
* Explainability
* Human Evaluation

---

# 14. Model Training Strategy

Phase 1

Train CNN Detector

---

Phase 2

Generate Event Dataset

---

Phase 3

Train Temporal Transformer

---

Phase 4

Build Retrieval Database

---

Phase 5

Implement RATM

---

Phase 6

Generate Clinical Insights

---

# 15. Future Model Evolution

Future versions may include:

## Audio Foundation Models

* Wav2Vec2
* Audio Spectrogram Transformer
* BEATs

---

## Multi-Modal Learning

Audio + Environment

---

## Self-Supervised Learning

Large-scale respiratory representation learning

---

## Personalized Respiratory Foundation Models

Patient-specific adaptation

---

# AI Architecture Summary

PRISM uses a hierarchical intelligence architecture.

```text
Audio
   │
   ▼
Mel Spectrogram
   │
   ▼
CNN Detector
   │
   ▼
Event Generator
   │
   ▼
Temporal Transformer
   │
   ▼
Environmental Correlation
   │
   ▼
RATM Engine
   │
   ▼
Clinical Insight Generation
```

This architecture balances practicality, research depth, computational feasibility, explainability, and future scalability while remaining achievable within a student-led research project.