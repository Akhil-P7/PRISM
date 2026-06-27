import os

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset

from models.cough_detector.model import CoughDetector
from models.disease_classifier.classifier import DISEASE_CLASSES, DiseaseClassifierHead
from models.shared.checkpoint import load_checkpoint


class EvalDataset(Dataset):
    def __init__(self, df, features_dir):
        self.data = df.to_dict("records")
        self.features_dir = features_dir
        self.label_map = {name: i for i, name in enumerate(DISEASE_CLASSES)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        mel_path = os.path.join(self.features_dir, row["mel_path"])
        mel = np.load(mel_path)
        mel_tensor = torch.from_numpy(mel).float().unsqueeze(0)

        # Safe normalize per sample
        mean = mel_tensor.mean()
        std = mel_tensor.std().clamp(min=1e-6)
        mel_tensor = (mel_tensor - mean) / std

        return mel_tensor, self.label_map[row["label"]]


def collate_fn(batch):
    mels, labels = zip(*batch, strict=False)
    max_len = max([m.shape[2] for m in mels])
    padded_mels = []
    for m in mels:
        pad_amount = max_len - m.shape[2]
        padded_mels.append(torch.nn.functional.pad(m, (0, pad_amount)))
    return torch.stack(padded_mels), torch.tensor(labels, dtype=torch.long)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load data
    features_dir = "datasets/features"
    manifest_path = os.path.join(features_dir, "manifest.csv")
    df = pd.read_csv(manifest_path)
    df = df[(~df["is_silent"]) & (df["label"].isin(DISEASE_CLASSES))]

    # Balance evaluation: take up to 200 samples per class
    sampled_dfs = []
    for cls in DISEASE_CLASSES:
        cls_df = df[df["label"] == cls]
        n_samples = min(200, len(cls_df))
        if n_samples > 0:
            sampled_dfs.append(cls_df.sample(n=n_samples, random_state=42))

    eval_df = pd.concat(sampled_dfs).reset_index(drop=True)
    logger.info(f"Evaluating on {len(eval_df)} balanced samples...")

    dataset = EvalDataset(eval_df, features_dir)
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    # 2. Load models
    logger.info("Loading CoughDetector CNN...")
    cnn = CoughDetector(pretrained=False)
    load_checkpoint(
        "models/checkpoints/cough_detector_finetuned.pt", cnn, device=device
    )
    cnn = cnn.to(device)
    cnn.eval()

    logger.info("Loading trained Disease Classifier...")
    classifier = DiseaseClassifierHead(input_dim=512, num_classes=len(DISEASE_CLASSES))
    classifier.load_state_dict(
        torch.load(
            "models/checkpoints/disease_classifier_v1.pt",
            map_location=device,
            weights_only=True,
        )
    )
    classifier = classifier.to(device)
    classifier.eval()

    # 3. Evaluate
    all_preds = []
    all_labels = []

    logger.info("Running inference...")
    with torch.no_grad():
        for mels, labels in dataloader:
            mels = mels.to(device)
            embeddings = cnn.encode(mels)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            logits = classifier(embeddings)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. Report
    report = classification_report(
        all_labels, all_preds, target_names=DISEASE_CLASSES, zero_division=0
    )
    print("\n" + "=" * 50)
    print(" DISEASE CLASSIFIER EVALUATION RESULTS")
    print("=" * 50)
    print(report)


if __name__ == "__main__":
    main()
