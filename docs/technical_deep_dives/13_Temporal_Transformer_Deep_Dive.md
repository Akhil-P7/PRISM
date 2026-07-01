# PRISM: Temporal Transformer Pipeline Overview

This document explains the architecture, data strategy, implementation, training, and evaluation of the Temporal Transformer — the second major AI component in the PRISM system, responsible for tracking how a patient's coughing behavior evolves over time and predicting disease trajectory.

---

## 1. Why a Temporal Transformer?

The CNN Cough Detector (Sprint 3) answers a single question: **"Is this audio segment a cough?"** But clinically, a single cough detection is not very useful. What matters is the *pattern* — is the patient getting better? Worse? Are there alarming spikes?

The Temporal Transformer sits **downstream** of the CNN in the PRISM pipeline and solves the next question: **"Given 30 days of cough statistics, what is the patient's respiratory trajectory?"**

```text
PRISM AI Pipeline (where the Temporal Transformer fits)

Raw Audio
    |
    v
Mel Spectrogram (Audio Representation Layer)
    |
    v
CNN Cough Detector (Event Detection Layer) .............. Sprint 3 -- COMPLETE
    |
    v
Daily Cough Statistics (Event Aggregation)
    |
    v
Temporal Transformer (Temporal Intelligence Layer) ...... Sprint 4 -- THIS DOCUMENT
    |
    v
Trajectory Prediction (Stable / Improving / Increasing / Abnormal)
```

---

## 2. Why Not LSTM?

Traditional sequential models like LSTMs were considered but rejected for this task. LSTMs process sequences step-by-step (day 1, then day 2, then day 3...) which creates a **sequential bottleneck** — the model's understanding of day 1 must be compressed through a hidden state to reach day 30. This leads to:

- **Vanishing gradients** over long sequences
- **Weak long-range reasoning** (day 1 patterns can't easily influence day 30 predictions)
- **Limited interpretability** (the hidden state is a black box)

Transformers use **self-attention**, which allows every day to directly attend to every other day in parallel. Day 1 can directly influence day 30's representation without passing through 29 intermediate states. This is far more powerful for detecting temporal patterns like "a spike on day 5 followed by a gradual increase through days 15-25."

---

## 3. Model Architecture

The Temporal Transformer is a lightweight **encoder-only** Transformer (inspired by the encoder half of "Attention Is All You Need" by Vaswani et al., 2017). It does not use a decoder because this is a classification task, not a sequence generation task.

### Data Flow

```text
Input: (Batch, 30, 5) -- 30 days x 5 features per day
       |
       v
Input Projection: Linear(5, 128) + LayerNorm + ReLU
       |  Transforms 5 raw features into a 128-dimensional representation
       |  that the Transformer can process.
       v
+ Sinusoidal Positional Encoding
       |  Injects day-position information (day 0, day 1, ... day 29)
       |  so the model knows the temporal ordering.
       v
Transformer Encoder (3 layers x 4 attention heads)
       |  Each layer: Pre-LayerNorm -> Multi-Head Self-Attention -> Feed-Forward(256) -> GELU
       |  Every day attends to every other day, learning which
       |  temporal relationships matter for classification.
       v
Mean Pooling
       |  Averages all 30 day representations into a single
       |  128-dim vector summarizing the entire trajectory.
       v
Classification Head: LayerNorm -> Linear(128, 64) -> GELU -> Linear(64, 4)
       |
       v
Output: 4 logits (one per trajectory class)
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Positional Encoding** | Sinusoidal (fixed) | No learnable parameters needed. Works well for short sequences (30 days). Uses sine/cosine functions at different frequencies so the model can distinguish day positions. |
| **Pooling Strategy** | Mean pooling | More robust than CLS token for tabular time-series. Averages attention across all days rather than relying on a single special token. |
| **Normalization** | Pre-LN (`norm_first=True`) | LayerNorm is applied *before* attention and feed-forward, not after. This makes training significantly more stable and reduces the need for learning rate warmup. |
| **Activation** | GELU | Smoother than ReLU, standard in modern Transformers (used in BERT, GPT). |
| **Architecture** | Encoder-only | We're classifying a sequence, not generating one. No decoder needed. |

### Hyperparameters (from `configs/training.yaml`)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `d_model` | 128 | Internal dimension of the Transformer |
| `n_heads` | 4 | Number of attention heads (each sees 128/4 = 32 dims) |
| `n_layers` | 3 | Depth of the Transformer encoder |
| `d_ff` | 256 | Hidden size of the feed-forward network within each layer |
| `dropout` | 0.1 | Applied after positional encoding and within the classification head |
| `max_sequence_length` | 30 | 30-day monitoring window |
| `num_classes` | 4 | Stable, Improving, Increasing, Abnormal |
| **Total Parameters** | **407,236** | All trainable. Lightweight by design. |

### Training Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `optimizer` | AdamW | Adam with decoupled weight decay — standard for Transformers |
| `learning_rate` | 0.0005 | Lower than the CNN's 0.001 because Transformers are more sensitive to LR |
| `weight_decay` | 0.01 | L2 regularization applied through AdamW |
| `scheduler` | CosineAnnealingLR | Smoothly decays LR to near-zero over the training run |
| `batch_size` | 16 | Small batch because the dataset is small (1,400 training patients) |
| `epochs` | 100 (max) | Training stops early when validation loss plateaus |
| `early_stopping_patience` | 15 | Stops training if no improvement for 15 consecutive epochs |
| `gradient_clipping` | max_norm=1.0 | Prevents gradient explosions common in Transformer training |

---

## 4. The Data Challenge & Synthetic Data Strategy

### The Problem

Our primary dataset (COUGHVID) is **cross-sectional** — each patient contributed a single recording session. There is no longitudinal data showing how the same patient's coughs evolved over 30 days. Real longitudinal cough monitoring datasets from wearable devices are not publicly available.

### The Solution: Synthetic Temporal Data

We generated **2,000 synthetic patients** (500 per trajectory class) with controlled, clinically-inspired statistical patterns. Each patient has exactly 30 days of data, with 5 features computed per day.

### Input Features (5 per day)

| Feature | Description | How It's Computed |
|---------|-------------|-------------------|
| `cough_count` | Number of cough events in a day | Poisson-distributed with class-specific lambda |
| `avg_duration` | Average cough duration (seconds) | Normal distribution, range 0.2-1.5s |
| `avg_intensity` | Average RMS energy of cough events | Normal distribution, range 0.1-0.95 |
| `night_ratio` | Fraction of coughs between 10 PM-6 AM | Beta/Normal distribution, range 0.0-1.0 |
| `inter_cough_interval` | Average seconds between consecutive coughs | Inversely derived from cough_count |

### Trajectory Generation Patterns

**Stable (Class 0):** Cough count hovers around a fixed baseline (~8 per day) with random Poisson noise. All other features remain flat. This simulates a patient with a chronic but non-worsening condition.

**Improving (Class 1):** Cough count linearly decreases from ~15 to ~4 over 30 days. Duration, intensity, and night ratio all decrease proportionally. This simulates a patient recovering from an illness.

**Increasing (Class 2):** The inverse of Improving. Cough count rises from ~4 to ~15. Duration, intensity, and night ratio all increase. This simulates a patient whose condition is worsening.

**Abnormal (Class 3):** Baseline cough count of ~5 with 3-6 random high-magnitude spikes (20-35 coughs per day) injected on random days. Spike days also show elevated duration (~1.1s), intensity (~0.8), and night ratio (~0.7). This simulates an irregular pattern that doesn't follow any smooth trend — the hallmark of a potentially dangerous condition.

### Data Split

| Split | Patients | Rows (patients x 30 days) | Purpose |
|-------|----------|---------------------------|---------|
| Train | 1,400 | 42,000 | Model training |
| Val | 300 | 9,000 | Early stopping & checkpoint selection |
| Test | 300 | 9,000 | Final held-out evaluation |

Splitting was done at the **patient level** (no patient appears in multiple splits), consistent with our CNN training strategy to prevent data leakage.

### Feature Normalization

All 5 features are **z-score normalized** before being fed to the model. Normalization statistics (mean, std) are computed exclusively from the training set and applied identically to the validation and test sets. This prevents information leakage from evaluation data.

---

## 5. Training Results

- **Training Environment:** Google Colab (T4 GPU)
- **Training Time:** ~50 seconds (40 epochs before early stopping)
- **Best Checkpoint Epoch:** 25

### Training Progression

The model converged extremely rapidly:

| Epoch | Train Loss | Train Acc | Train F1 | Val Loss | Val Acc | Val F1 |
|-------|-----------|-----------|----------|----------|---------|--------|
| 1 | 0.3268 | 0.902 | 0.902 | 0.0104 | 1.000 | 1.000 |
| 2 | 0.0075 | 0.999 | 0.999 | 0.0016 | 1.000 | 1.000 |
| 6 | 0.0005 | 1.000 | 1.000 | 0.0002 | 1.000 | 1.000 |
| 25 | 0.0000 | 1.000 | 1.000 | 0.0000 | 1.000 | 1.000 |
| 40 | *early stop* | | | | | |

The model achieved perfect validation accuracy by Epoch 1 and continued reducing loss until early stopping at Epoch 40. The best checkpoint (lowest validation loss) was saved at Epoch 25.

> **Why is accuracy so high?** This is expected behavior on synthetic data with clearly separable trajectory patterns. The mathematical rules we used to generate data (linear trends, Poisson spikes) produce patterns that are unambiguous for a Transformer to learn. This is by design — the goal of this sprint was to prove the architecture works. When real longitudinal clinical data becomes available, accuracy will naturally be lower and the model can be retrained on the same infrastructure without any code changes.

---

## 6. Test Set Evaluation (Epoch 25 Checkpoint)

### Aggregate Metrics

| Metric | Value |
|--------|-------|
| **Loss** | 0.00001 |
| **Accuracy** | 100.0% |
| **Macro F1** | 1.0000 |
| **Weighted F1** | 1.0000 |
| **Macro Precision** | 1.0000 |
| **Macro Recall** | 1.0000 |

### Per-Class Performance

| Trajectory | Precision | Recall | F1-Score | Test Samples |
|------------|-----------|--------|----------|--------------|
| Stable | 1.0000 | 1.0000 | 1.0000 | 75 |
| Improving | 1.0000 | 1.0000 | 1.0000 | 71 |
| Increasing | 1.0000 | 1.0000 | 1.0000 | 80 |
| Abnormal | 1.0000 | 1.0000 | 1.0000 | 74 |

### Confusion Matrix

```text
                 Predicted
              Stable  Improving  Increasing  Abnormal
Actual
  Stable         75          0           0         0
  Improving       0         71           0         0
  Increasing      0          0          80         0
  Abnormal        0          0           0        74
```

A perfect diagonal matrix — zero misclassifications across all 300 test patients. Verified both on Colab (T4 GPU) and locally (CPU), confirming the checkpoint is portable and produces identical results.

---

## 7. Code Organization

### Model Files

| File | Purpose |
|------|---------|
| [`model.py`](../models/temporal_transformer/model.py) | `TemporalTransformer` and `SinusoidalPositionalEncoding` classes |
| [`dataset.py`](../models/temporal_transformer/dataset.py) | `TemporalDataset` (PyTorch Dataset) and `create_temporal_dataloaders()` factory |
| [`generate_temporal_data.py`](../models/temporal_transformer/generate_temporal_data.py) | Synthetic 30-day timeline generator with 4 trajectory patterns |
| [`train.py`](../models/temporal_transformer/train.py) | `TemporalTrainer` with AdamW, cosine annealing, gradient clipping, early stopping |
| [`run_training.py`](../models/temporal_transformer/run_training.py) | CLI entry point (`--dry-run`, `--generate-data`, `--epochs`, `--batch-size`) |
| [`evaluate.py`](../models/temporal_transformer/evaluate.py) | Test-set evaluation with per-class metrics, confusion matrix, JSON output |

### Shared Utilities

| File | What Was Added |
|------|----------------|
| [`metrics.py`](../models/shared/metrics.py) | `MultiClassMetricTracker` — accumulates 4-class predictions across batches, computes macro/weighted F1 |
| [`checkpoint.py`](../models/shared/checkpoint.py) | Reused as-is from the CNN pipeline for `save_checkpoint()` and `load_checkpoint()` |

### Configuration

| File | Relevant Section |
|------|-----------------|
| [`training.yaml`](../configs/training.yaml) | `temporal_transformer:` block (lines 41-54) — all hyperparameters |

### Generated Data

| File | Contents |
|------|----------|
| [`datasets/temporal/temporal_train.csv`](../datasets/temporal/temporal_train.csv) | 1,400 patients x 30 days = 42,000 rows |
| [`datasets/temporal/temporal_val.csv`](../datasets/temporal/temporal_val.csv) | 300 patients x 30 days = 9,000 rows |
| [`datasets/temporal/temporal_test.csv`](../datasets/temporal/temporal_test.csv) | 300 patients x 30 days = 9,000 rows |

### Artifacts

| File | Description |
|------|-------------|
| [`temporal_transformer_best.pt`](../models/checkpoints/temporal_transformer_best.pt) | Best model checkpoint (Epoch 25) |
| [`temporal_eval.json`](../models/checkpoints/temporal_eval.json) | Colab evaluation results |
| [`temporal_eval.json`](../evaluation/temporal_analysis/temporal_eval.json) | Local evaluation results (identical to Colab) |
| [`PRISM_Temporal_Training.ipynb`](../PRISM_Temporal_Training.ipynb) | Google Colab training notebook |

---

## 8. How This Connects to the Full PRISM Pipeline

The Temporal Transformer is one of three core AI components in PRISM:

```text
1. CNN Cough Detector (Sprint 3) ........... Audio -> Cough Detection
        |
        |--- Embeddings (512-dim) -------> TurboVec Retrieval (Sprint 5)
        |
        |--- Cough Predictions ----------> Daily Statistics Aggregation
                                                    |
                                                    v
2. Temporal Transformer (Sprint 4) ........ 30-Day Stats -> Trajectory Prediction
                                                    |
                                                    v
3. RATM Engine (Sprint 5+) ................ Trajectory + Similar Cases -> Clinical Insight
```

In a production deployment, the flow would be:
1. Patient's audio is recorded continuously over 30 days
2. The CNN processes each audio segment, detecting coughs and computing statistics
3. Daily statistics (cough_count, avg_duration, avg_intensity, night_ratio, inter_cough_interval) are aggregated
4. The Temporal Transformer takes the 30-day feature matrix and predicts the trajectory class
5. The RATM engine retrieves similar historical cases and generates clinical explanations

---

## 9. Limitations & Future Work

1. **Synthetic Data:** The current model was trained entirely on synthetic data with mathematically clean patterns. Real patient data will contain far more noise, missing days, variable-length sequences, and ambiguous trajectory boundaries. The model architecture and training infrastructure are production-ready — only the data needs to change.

2. **Variable-Length Sequences:** The model already supports padding masks for sequences shorter than 30 days (via the `padding_mask` parameter in `forward()`), but this hasn't been exercised yet since all synthetic patients have exactly 30 days.

3. **Attention Visualization:** The self-attention weights from the Transformer encoder could be extracted and visualized to show *which days* the model considers most important for each trajectory prediction — providing clinical interpretability.

4. **Integration with Real CNN Outputs:** Once the full pipeline is wired together, the 5 daily features will be computed from actual CNN inference results rather than synthetic generation. This is a straightforward aggregation step that doesn't require model changes.
