"""
PRISM Scripts — Embedding & Retrieval Verification

Sanity-check script that:
1. Loads the CNN checkpoint and a few test segments
2. Extracts embeddings on CPU
3. Queries the TurboVec index for nearest neighbours
4. Prints a formatted table showing semantic coherence
   (cough queries should retrieve other cough segments)

Usage::

    python scripts/verify_embeddings.py
    python scripts/verify_embeddings.py --num-samples 10
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table

from models.cough_detector.model import CoughDetector
from models.shared.checkpoint import load_checkpoint
from models.shared.transforms import Normalize
from retrieval.vector_store.search import TurboVecSearchEngine

console = Console()


def verify(
    checkpoint_path: str = "models/checkpoints/cough_detector_best.pt",
    manifest_path: str = "datasets/features/manifest.csv",
    features_dir: str = "datasets/features",
    index_path: str = "retrieval/vector_store/cough_embeddings.tq",
    metadata_path: str = "retrieval/vector_store/index_metadata.csv",
    num_samples: int = 5,
    top_k: int = 3,
    seed: int = 42,
) -> None:
    """Run the full verification pipeline."""
    random.seed(seed)
    np.random.seed(seed)

    # ---- Load model ----
    console.rule("[bold cyan]Step 1: Load CNN Checkpoint")
    model = CoughDetector(pretrained=False)
    info = load_checkpoint(checkpoint_path, model, device=torch.device("cpu"))
    model.eval()
    console.print(f"  Loaded epoch {info['epoch']} | Metrics: {info['metrics']}")

    # ---- Load manifest and pick samples ----
    console.rule("[bold cyan]Step 2: Select Test Samples")
    df = pd.read_csv(manifest_path)
    df_active = df[~df["is_silent"]].copy().reset_index(drop=True)

    # Try to pick a mix of cough and non-cough segments
    cough_df = df_active[df_active["is_cough"] == True]  # noqa: E712
    non_cough_df = df_active[df_active["is_cough"] == False]  # noqa: E712

    n_cough = min(num_samples // 2 + 1, len(cough_df))
    n_non_cough = min(num_samples - n_cough, len(non_cough_df))

    sample_indices = random.sample(range(len(cough_df)), n_cough) + random.sample(
        range(len(non_cough_df)), n_non_cough
    )
    samples = pd.concat(
        [
            cough_df.iloc[sample_indices[:n_cough]],
            non_cough_df.iloc[sample_indices[n_cough:]],
        ]
    ).reset_index(drop=True)

    console.print(
        f"  Selected {len(samples)} samples ({n_cough} cough, {n_non_cough} non-cough)"
    )

    # ---- Extract embeddings for samples ----
    console.rule("[bold cyan]Step 3: Extract Embeddings (CPU)")
    normalize = Normalize()
    query_embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for _, row in samples.iterrows():
            mel_path = Path(features_dir) / row["mel_path"]
            mel = np.load(str(mel_path))
            mel = (
                torch.from_numpy(mel).float().unsqueeze(0).unsqueeze(0)
            )  # (1, 1, 128, T)
            mel = normalize(mel)
            features = model.encode(mel)  # (1, 512)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            query_embeddings.append(features.numpy()[0])

    console.print(f"  Extracted {len(query_embeddings)} embeddings")

    # ---- Load search engine ----
    console.rule("[bold cyan]Step 4: Query TurboVec Index")

    if not Path(index_path).exists():
        console.print(
            "[bold red]  Index not found![/] Run the index builder first:\n"
            "  python -m retrieval.vector_store.index_builder"
        )
        return

    engine = TurboVecSearchEngine(index_path=index_path, metadata_path=metadata_path)

    # ---- Search and display results ----
    console.rule("[bold cyan]Step 5: Results")

    coherence_hits = 0
    coherence_total = 0

    for i, (_, row) in enumerate(samples.iterrows()):
        query_label = "🔴 COUGH" if row["is_cough"] else "🟢 Non-cough"
        query_subject = row["subject_id"]

        table = Table(
            title=f"Query {i + 1}: {query_label} | Subject: {query_subject}",
            show_lines=True,
        )
        table.add_column("Rank", style="dim", width=5)
        table.add_column("Similarity", style="bold", width=12)
        table.add_column("Type", width=14)
        table.add_column("Subject", width=15)
        table.add_column("Recording", width=15)
        table.add_column("Match?", width=8)

        results = engine.search(query_embeddings[i], k=top_k)

        for rank, r in enumerate(results, 1):
            result_label = "🔴 COUGH" if r.is_cough else "🟢 Non-cough"
            is_coherent = r.is_cough == row["is_cough"]
            match_icon = "✅" if is_coherent else "❌"

            if is_coherent:
                coherence_hits += 1
            coherence_total += 1

            table.add_row(
                str(rank),
                f"{r.similarity_score:.4f}",
                result_label,
                r.subject_id,
                r.recording_id,
                match_icon,
            )

        console.print(table)
        console.print()

    # ---- Summary ----
    console.rule("[bold cyan]Summary")
    coherence_pct = (
        (coherence_hits / coherence_total * 100) if coherence_total > 0 else 0
    )
    color = (
        "green" if coherence_pct >= 70 else "yellow" if coherence_pct >= 50 else "red"
    )
    console.print(
        f"  Semantic coherence: [{color}]{coherence_hits}/{coherence_total} "
        f"({coherence_pct:.1f}%)[/{color}]"
    )
    console.print(
        "  (Cough queries retrieving cough results, non-cough retrieving non-cough)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PRISM — Verify embedding extraction and retrieval quality."
    )
    parser.add_argument(
        "--num-samples", type=int, default=5, help="Number of test samples."
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top-K results per query.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    verify(num_samples=args.num_samples, top_k=args.top_k, seed=args.seed)


if __name__ == "__main__":
    main()
