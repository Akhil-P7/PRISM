"""PRISM Models — Embedding Generation

Provides the embedding extraction pipeline that uses the trained
CoughDetector backbone to produce 512-dim L2-normalised vectors.
"""

from models.embeddings.extract_embeddings import extract_embeddings

__all__ = ["extract_embeddings"]
