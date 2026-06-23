# PRISM Cough Detector: CNN Implementation & Working Guide

## 1. Overview
The Cough Detector is a convolutional neural network (CNN) designed to identify cough events from pediatric respiratory audio recordings. It operates on **Mel Spectrograms** (visual representations of audio frequencies over time) and serves two primary purposes:
1. **Binary Classification**: Determines if a 3-second audio segment contains a cough (`is_cough = True/False`).
2. **Embedding Generation**: Compresses the audio segment into a rich 512-dimensional vector. These vectors will later be fed into **TurboVec** for high-speed similarity search (finding similar historical patient cases).

---

## 2. Model Architecture
The model is implemented in PyTorch (`models/cough_detector/model.py`) and utilizes a modified **ResNet-18** architecture.

### Backbone Adaptation
Standard ResNet-18 is designed for RGB images (3 color channels). Respiratory audio is single-channel (mono).
- **The Fix**: We modified the first convolutional layer (`conv1`) to accept **1 input channel** instead of 3.
- **Transfer Learning**: To retain the benefits of ImageNet pretraining, the weights of the original 3 channels were averaged into the new single channel. This gives the model a massive head start in recognizing edges and textures in the spectrograms.

### Dual-Head Output
After the ResNet backbone extracts features, a Global Average Pooling layer flattens the output into a 512-dimensional tensor. This tensor branches into two specialized heads:
1. **Cough Head (`fc_cough`)**: A simple linear layer that outputs a single logit. This is passed through a Sigmoid function to give a probability (0.0 to 1.0) of a cough being present.
2. **Embedding Head (`fc_embed`)**: A multi-layer perceptron (Linear → BatchNorm → ReLU → Linear) that projects the features into a 512-dim vector. We apply **L2-Normalization** to this vector, which maps it onto a unit hypersphere—the mathematical requirement for efficient Cosine Similarity searches in TurboVec.

---

## 3. Data Pipeline & DataLoader

The pipeline (`models/cough_detector/dataset.py`) converts the 131,000 `.npy` files created in Sprint 2 into training batches.

> [!IMPORTANT]
> **Subject-Level Splitting (Zero Data Leakage)**
> Traditional 80/20 random splits are dangerous in medical AI because audio from the *same patient* might end up in both the training and testing sets, causing the model to "memorize" the patient's background noise rather than learning to detect coughs. We implemented strict **subject-based splitting**, guaranteeing that a patient in the Test set has never been seen by the model during Training.

### Class Imbalance Handling
The dataset has roughly 85,000 coughs and 46,000 non-coughs. To prevent the model from becoming biased toward the majority class:
1. **Weighted Random Sampler**: The PyTorch `DataLoader` calculates the frequency of each class and assigns a higher sampling probability to the minority class.
2. **Loss Weighting**: The `BCEWithLogitsLoss` function is supplied with a `pos_weight` parameter, forcing the model to pay heavier penalties for misclassifying the minority class.

### On-the-Fly Augmentation (SpecAugment)
To prevent overfitting and force the model to generalize, we apply **SpecAugment** (`models/shared/transforms.py`) dynamically to the training data in memory:
- **Time Masking**: Randomly zeros out vertical blocks of time (simulates audio dropping out).
- **Frequency Masking**: Randomly zeros out horizontal bands of frequencies (simulates different microphone hardware constraints).
- **Gaussian Noise**: Adds slight background static.

---

## 4. Training Strategy

The training loop (`models/cough_detector/train.py`) is designed for stability and convergence:

- **Optimizer**: `Adam` with a base learning rate of `0.001`.
- **Scheduler**: `CosineAnnealingLR` gradually reduces the learning rate in a cosine curve over 50 epochs, allowing the model to make large adjustments early on, and fine-tune its weights as it gets closer to the optimal solution.
- **Early Stopping**: The trainer monitors the Validation Loss. If the model fails to improve for 7 consecutive epochs (`patience=7`), training halts automatically to prevent overfitting.
- **Metrics Tracking**: Alongside Loss, we track **AUC-ROC** (Area Under the Receiver Operating Characteristic Curve), **F1-Score**, Precision, and Recall. AUC-ROC is the primary metric, as it handles imbalanced classes much better than raw accuracy.

---

## 5. Performance & Hardware Considerations

### The CPU/Disk Bottleneck
Because your system utilizes an **Intel UHD 620 integrated GPU**, PyTorch falls back to the CPU for training.

By default, PyTorch attempts to spawn multiple background processes (`num_workers=4`) to load data from the hard drive while the CPU trains the network. However, on Windows, PyTorch multiprocessing can frequently cause deadlocks (hanging indefinitely).

To ensure stability, we are currently utilizing `num_workers=0` (loading data on the main thread). Because the CPU must wait for the hard drive to load 131,000 files sequentially per epoch, training is constrained by **Disk I/O speeds**, not CPU computation.

To mitigate this while retaining data integrity, the batch size was increased to `128` to drastically reduce the Python looping overhead.
