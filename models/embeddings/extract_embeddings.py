"""
PRISM Models — Embedding Extraction

Loads the trained CoughDetector checkpoint and extracts 512-dim embeddings
for every non-silent segment in the dataset.  Outputs:

    embeddings_matrix.npy    — shape (N, 512), float32
    embeddings_metadata.csv  — row-aligned metadata (segment_id, subject_id, …)

Designed to run on Google Colab (GPU), but works on CPU too.

Usage::

    python -m models.embeddings.extract_embeddings \
        --checkpoint models/checkpoints/cough_detector_best.pt \
        --manifest datasets/features/manifest.csv \
        --features-dir datasets/features \
        --output-dir models/embeddings \
        --batch-size 256
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.cough_detector.model import CoughDetector
from models.shared.checkpoint import load_checkpoint
from models.shared.transforms import Normalize

# ---------------------------------------------------------------------------
# Dataset — minimal, no augmentation, just load + normalise
# ---------------------------------------------------------------------------


class EmbeddingDataset(Dataset):
    """
    Lightweight dataset for embedding extraction.

    No augmentation, no label logic — just loads mel spectrograms
    and normalises them for inference.
    """

    def __init__(self, df: pd.DataFrame, features_dir: str | Path) -> None:
        self.df = df.reset_index(drop=True)
        self.features_dir = Path(features_dir)
        self.normalize = Normalize()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = self.df.iloc[idx]
        mel_path = self.features_dir / row["mel_path"]
        mel = np.load(str(mel_path))  # (128, T)
        mel = torch.from_numpy(mel).float().unsqueeze(0)  # (1, 128, T)
        mel = self.normalize(mel)
        return mel


# ---------------------------------------------------------------------------
# Extraction logic
# ---------------------------------------------------------------------------


def extract_embeddings(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    features_dir: str | Path,
    output_dir: str | Path,
    batch_size: int = 256,
    num_workers: int = 4,
    device: str = "auto",
) -> tuple[Path, Path]:
    """
    Extract embeddings for all non-silent segments.

    Args:
        checkpoint_path: path to cough_detector_best.pt
        manifest_path: path to manifest.csv
        features_dir: root dir containing mel/ folder
        output_dir: where to write the output files
        batch_size: inference batch size
        num_workers: DataLoader workers
        device: 'auto', 'cuda', 'cpu', or 'mps'

    Returns:
        Tuple of (matrix_path, metadata_path).
    """
    # ---- Resolve device ----
    if device == "auto":
        if torch.cuda.is_available():
            device_obj = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_obj = torch.device("mps")
        else:
            device_obj = torch.device("cpu")
    else:
        device_obj = torch.device(device)
    logger.info(f"Using device: {device_obj}")

    # ---- Load model ----
    model = CoughDetector(pretrained=False)
    info = load_checkpoint(checkpoint_path, model, device=device_obj)
    model.to(device_obj)
    model.eval()
    logger.info(
        f"Loaded checkpoint from epoch {info['epoch']} | metrics: {info['metrics']}"
    )

    # ---- Prepare data ----
    df = pd.read_csv(manifest_path)
    df_active = df[~df["is_silent"]].copy().reset_index(drop=True)
    logger.info(f"Total segments: {len(df)} | Non-silent: {len(df_active)}")

    dataset = EmbeddingDataset(df_active, features_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # CRITICAL: must preserve row order for metadata alignment
        num_workers=num_workers,
        pin_memory=(device_obj.type == "cuda"),
        drop_last=False,
    )

    # ---- Extract embeddings ----
    all_embeddings: list[np.ndarray] = []
    total_segments = len(df_active)

    logger.info(
        f"Extracting embeddings for {total_segments} segments (batch_size={batch_size})..."
    )

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings", unit="batch"):
            batch = batch.to(device_obj)
            features = model.encode(batch)  # (B, 512)
            # L2-normalise for cosine similarity downstream
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            all_embeddings.append(features.cpu().numpy())

    # ---- Concatenate and save ----
    embeddings_matrix = np.concatenate(all_embeddings, axis=0)  # (N, 512)
    assert embeddings_matrix.shape[0] == total_segments, (
        f"Mismatch: matrix has {embeddings_matrix.shape[0]} rows, "
        f"expected {total_segments}"
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = output_dir / "embeddings_matrix.npy"
    np.save(str(matrix_path), embeddings_matrix)
    matrix_mb = matrix_path.stat().st_size / (1024 * 1024)
    logger.success(
        f"Saved embeddings matrix: {matrix_path} | shape={embeddings_matrix.shape} | {matrix_mb:.1f} MB"
    )

    # ---- Save metadata (row-aligned with the matrix) ----
    metadata_cols = ["subject_id", "recording_id", "mel_path", "is_cough"]
    # Add segment_id if present, otherwise create one
    if "segment_id" in df_active.columns:
        metadata_cols.insert(0, "segment_id")
    else:
        df_active = df_active.copy()
        df_active.insert(0, "segment_id", range(len(df_active)))
        metadata_cols.insert(0, "segment_id")

    metadata_df = df_active[metadata_cols].copy()
    metadata_df.insert(0, "embedding_idx", range(len(metadata_df)))

    metadata_path = output_dir / "embeddings_metadata.csv"
    metadata_df.to_csv(str(metadata_path), index=False)
    logger.success(f"Saved metadata: {metadata_path} | {len(metadata_df)} rows")

    return matrix_path, metadata_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PRISM — Extract 512-dim embeddings from the trained CNN checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/cough_detector_best.pt",
        help="Path to the trained CNN checkpoint.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="datasets/features/manifest.csv",
        help="Path to manifest.csv.",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default="datasets/features",
        help="Root directory containing mel/ folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/embeddings",
        help="Directory to write embeddings_matrix.npy and embeddings_metadata.csv.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Inference batch size (default 256).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count (default 4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to run inference on (default: auto-detect).",
    )
    args = parser.parse_args()

    extract_embeddings(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
