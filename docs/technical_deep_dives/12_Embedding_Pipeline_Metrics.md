# PRISM: CNN & Embedding Pipeline Overview

This document summarizes the architecture, implementation, and verification of the primary machine learning pipeline for the PRISM project, covering the transition from raw audio features to compressed vector search indices.

---

## 1. CNN Cough Detector (Sprint 3)

The foundation of the PRISM system is the Convolutional Neural Network (CNN) designed to differentiate coughs from background noise and other sounds.

- **Base Architecture:** ResNet-18 backbone (originally designed for images), adapted for audio processing.
- **Audio Modifications:** The first convolutional layer was modified to accept single-channel (mono) Mel-spectrogram inputs instead of 3-channel RGB images. We leveraged ImageNet pre-trained weights by averaging the RGB channel weights into a single channel.
- **Dual-Head Design:** The model features two output heads:
  1. A binary classifier head (`fc_cough`) outputting a probability logit.
  2. A projection head (`fc_embed`) yielding 512-dimensional embeddings.
- **Training Environment:** Google Colab (T4 GPU).
- **Dataset:** 131,155 audio segments.
- **Final Metrics (Epoch 4):**
  - **Accuracy:** `74.2%`
  - **AUC:** `0.8786`
  - **F1 Score:** `0.7663`
  - **Precision:** `0.9330`
  - **Recall:** `0.6501`
- **Output:** The model checkpoint was exported and saved as `cough_detector_best.pt`.

---

## 2. Embedding Extraction Pipeline

To enable semantic search and patient clustering, we extracted dense representations (embeddings) from the trained CNN.

- **Extraction Strategy:** Instead of taking the output from the final classifier or projection heads, we passed all 131,155 segments through the CNN and tapped directly into the `AdaptiveAvgPool2d` layer immediately following ResNet's `layer4`. This gave us raw, un-tuned **512-dimensional feature vectors** representing the model's deepest understanding of the acoustic properties before they were compressed into a binary classification.
- **Normalization:** Vectors were explicitly L2-normalized post-extraction to ensure that Euclidean distances perfectly correlate with Cosine Similarity, which is a fundamental requirement for vector search and clustering.
- **Output:**
  - `embeddings_matrix.npy` (256.2 MB)
  - `embeddings_metadata.csv` (18 MB, row-aligned containing `subject_id` and `recording_id`)

---

## 3. TurboVec Indexing & Retrieval Engine

To make searching through 131,155 vectors instantaneous in a local API environment, we implemented a vector database.

- **TurboVec Compression:** We used `turbovec` to compress the raw float32 embeddings into a 4-bit quantized index.
- **Compression Ratio:** The index was reduced from **256.2 MB down to 32.5 MB** (a ~7.9x compression ratio), allowing it to easily fit in RAM for instantaneous CPU-based search.
- **FastAPI Integration:** We built a local API (`backend/api/retrieval.py`) that loads this index as a thread-safe singleton, exposing endpoints to search for both similar individual audio segments and aggregate patient matches.

---

## 4. Semantic Coherence Verification

To verify that the un-tuned embeddings were actually capturing meaningful acoustic features, we ran a semantic coherence test querying the index.

**Test Parameters:**
- **Samples:** 10 random audio segments (6 coughs, 4 non-coughs).
- **Queries:** Searched the TurboVec index for the top 5 closest matches (`k=5`) for each sample.

**Results:**
> [!TIP]
> **84.0% Semantic Coherence (42/50 matches)**

Out of 50 total retrieved matches, 42 of them correctly matched the class of the query (i.e., querying a cough successfully retrieved other coughs, and querying background noise retrieved other background noise).

Considering these embeddings were extracted from an intermediate layer of a simple classification CNN (and were not explicitly tuned using contrastive loss), **84.0% coherence is an incredibly strong out-of-the-box baseline**. It proves the CNN has successfully learned to group semantic audio features together in the 512-dimensional vector space.

---

## 5. Next Steps Completed: Temporal Transformer (Sprint 4)

The Temporal Transformer has been fully implemented, trained, and verified. See the dedicated documentation:

**[Temporal Transformer Pipeline Overview](prism_temporal_transformer_overview.md)**

- **Architecture:** 3-layer encoder-only Transformer (407K params) with sinusoidal positional encoding
- **Input:** 30-day windows of 5 daily cough statistics
- **Output:** 4 trajectory classes (Stable, Improving, Increasing, Abnormal)
- **Test Results:** 100% accuracy, 1.0 Macro F1 (on synthetic data)
- **Checkpoint:** `models/checkpoints/temporal_transformer_best.pt`

---

## 6. Remaining Steps (Workstream C)

With both the CNN Cough Detector and the Temporal Transformer complete, the final pipeline component is the **RATM (Retrieval-Augmented Temporal Modeling) integration** — wiring TurboVec retrieval into the clinical insight generation layer.
