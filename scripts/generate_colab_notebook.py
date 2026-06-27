import nbformat as nbf

nb = nbf.v4.new_notebook()

c1 = nbf.v4.new_code_cell(
    "# 1. Mount Google Drive\nfrom google.colab import drive\ndrive.mount('/content/drive')"
)

c2 = nbf.v4.new_code_cell(
    "# 2. Install dependencies\n!pip install loguru torch pandas numpy"
)

c3 = nbf.v4.new_code_cell(
    """# 3. Extract datasets and setup python path
import os
import sys
import zipfile

# Assuming the ZIP and prism-colab are in the root of your Google Drive
DRIVE_ROOT = '/content/drive/MyDrive'
PRISM_COLAB_DIR = os.path.join(DRIVE_ROOT, 'prism-colab')
ZIP_PATH = os.path.join(DRIVE_ROOT, 'datasets-features.zip')
EXTRACT_DIR = '/content/datasets'

# Add prism-colab to system path so we can import the models
if PRISM_COLAB_DIR not in sys.path:
    sys.path.append(PRISM_COLAB_DIR)

# Unzip features locally to the colab instance for faster I/O
if not os.path.exists(EXTRACT_DIR):
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    if os.path.exists(ZIP_PATH):
        print(f"Extracting {ZIP_PATH}...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print("Extraction complete.")
    else:
        print(f"ZIP file not found at {ZIP_PATH}")
"""
)

c4 = nbf.v4.new_code_cell(
    """# 4. Define Unified Dataset Loader
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

DISEASE_CLASSES = ["Healthy", "COVID-19", "COPD", "Asthma", "Pneumonia", "URTI", "LRTI", "Bronchiectasis", "Bronchiolitis"]
NUM_CLASSES = len(DISEASE_CLASSES)

class UnifiedEmbeddingDataset(Dataset):
    def __init__(self, features_dir, manifest_path):
        self.features_dir = features_dir
        df = pd.read_csv(manifest_path)
        # Filter for rows that are not silent and have a known disease label
        df = df[(df['is_silent'] == False) & (df['label'].isin(DISEASE_CLASSES))]
        self.data = df.to_dict('records')
        self.label_map = {name: i for i, name in enumerate(DISEASE_CLASSES)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        mel_path = os.path.join(self.features_dir, row['mel_path'])
        mel = np.load(mel_path)
        # Convert to tensor (1, 128, T)
        mel_tensor = torch.from_numpy(mel).float().unsqueeze(0)

        # Normalize per-sample
        mean = mel_tensor.mean()
        std = mel_tensor.std().clamp(min=1e-6)
        mel_tensor = (mel_tensor - mean) / std

        label_idx = self.label_map[row['label']]
        # Domain: 1.0 for smartphone (COUGHVID/Coswara), 0.0 for stethoscope (ICBHI)
        domain_idx = 0.0 if row['dataset'] == 'ICBHI' else 1.0

        return mel_tensor, label_idx, domain_idx

MANIFEST_PATH = os.path.join(EXTRACT_DIR, 'manifest.csv')
dataset = UnifiedEmbeddingDataset(EXTRACT_DIR, MANIFEST_PATH)

# Compute Class Weights to handle massive imbalance
from sklearn.utils.class_weight import compute_class_weight
y_train = [row['label'] for row in dataset.data]
class_weights_np = compute_class_weight('balanced', classes=np.array(DISEASE_CLASSES), y=y_train)
class_weights = torch.FloatTensor(class_weights_np).to('cuda' if torch.cuda.is_available() else 'cpu')

def collate_fn(batch):
    mels, labels, domains = zip(*batch)
    max_len = max([m.shape[2] for m in mels])
    padded_mels = []
    for m in mels:
        pad_amount = max_len - m.shape[2]
        padded_mels.append(torch.nn.functional.pad(m, (0, pad_amount)))
    return torch.stack(padded_mels), torch.tensor(labels, dtype=torch.long), torch.tensor(domains, dtype=torch.float32)

dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
print(f"Loaded dataset with {len(dataset)} valid segments.")
"""
)

c5 = nbf.v4.new_code_cell(
    """# 5. Training Loop
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from models.disease_classifier.classifier import UnifiedUniversalClassifier
from models.cough_detector.model import CoughDetector
from models.shared.checkpoint import load_checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load the pre-trained CNN to extract 512-D embeddings on the fly
logger.info("Loading pre-trained CoughDetector...")
cnn = CoughDetector(pretrained=False)
checkpoint_path = os.path.join(PRISM_COLAB_DIR, 'models', 'checkpoints', 'cough_detector_finetuned.pt')
load_checkpoint(checkpoint_path, cnn, device=device)
cnn = cnn.to(device)
cnn.eval()

# 2. Initialize the new Disease Classifier
model = UnifiedUniversalClassifier(input_dim=512, num_classes=NUM_CLASSES)
model = model.to(device)

criterion_disease = nn.CrossEntropyLoss(weight=class_weights)
criterion_domain = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 25

logger.info(f"Starting Domain Adversarial Training on {device}")
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    p = float(epoch) / epochs
    alpha = 2. / (1. + torch.exp(torch.tensor(-10. * p)).item()) - 1.

    for mels, labels, domains in dataloader:
        mels = mels.to(device)
        labels = labels.to(device)
        domains = domains.to(device)

        # Extract embeddings
        with torch.no_grad():
            embeddings = cnn.encode(mels)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        optimizer.zero_grad()
        disease_logits, domain_logits = model(embeddings, alpha=alpha)

        loss_disease = criterion_disease(disease_logits, labels)
        loss_domain = criterion_domain(domain_logits.squeeze(-1), domains)
        loss = loss_disease + loss_domain

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    logger.info(f"Epoch {epoch}/{epochs} | Loss={total_loss/len(dataloader):.4f}")

SAVE_PATH = os.path.join(PRISM_COLAB_DIR, 'disease_classifier_v1.pt')
torch.save(model.classifier.state_dict(), SAVE_PATH)
logger.info(f"Weights successfully saved to {SAVE_PATH}")
"""
)

nb["cells"] = [c1, c2, c3, c4, c5]
with open("c:/Users/Dell/Desktop/PRISM/PRISM_Disease_Classification.ipynb", "w") as f:
    nbf.write(nb, f)
