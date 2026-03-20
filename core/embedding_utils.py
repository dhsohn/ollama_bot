"""Shared embedding utilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """Apply L2 normalization to each row vector."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / norms


def embed_texts(
    encoder: Any,
    texts: Sequence[str],
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Generate text embeddings with a ``TextEmbedding``-compatible encoder."""
    if encoder is None:
        raise RuntimeError("encoder is not initialized")

    payload = list(texts)
    if hasattr(encoder, "embed"):
        raw_vectors = list(encoder.embed(payload))
    else:
        raw_vectors = encoder.encode(payload, normalize_embeddings=False)

    vectors = np.asarray(raw_vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if normalize:
        vectors = normalize_rows(vectors)
    return vectors
