"""Embedding storage and numpy-based similarity search.

HyperRecall keeps embeddings as raw little-endian float32 BLOBs in SQLite and does
nearest-neighbour search in numpy. This keeps a HyperRecall database a single
portable file with no native extensions (no sqlite-vss) — see DESIGN.md.

The embedding *model* is pluggable. Out of the box we ship :func:`hash_embed`,
a deterministic, dependency-free hashing embedder. It is **not** semantic — it
exists so the whole pipeline runs and tests are reproducible. Swap in a real
model by passing an ``embed`` callable to :class:`hyperrecall.Mesh`.
"""

from __future__ import annotations

import hashlib
from typing import Callable

import numpy as np

EmbedFn = Callable[[str], np.ndarray]

DEFAULT_DIM = 256


def hash_embed(text: str, dim: int = DEFAULT_DIM) -> np.ndarray:
    """Deterministic hashing embedder.

    Tokens are hashed into buckets (signed hashing trick) and the resulting
    vector is L2-normalised. Same text -> identical vector, always. Good enough
    to make lexical overlap show up as cosine similarity for demos and tests;
    replace with a real model for production semantics.
    """
    vec = np.zeros(dim, dtype=np.float32)
    tokens = _tokenize(text)
    if not tokens:
        return vec
    for tok in tokens:
        h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        sign = 1.0 if (h[4] & 1) else -1.0
        vec[idx] += sign
    return _l2_normalize(vec)


def _tokenize(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return (vec / norm).astype(np.float32)


def to_blob(vec: np.ndarray) -> bytes:
    """Serialize a vector to a little-endian float32 blob."""
    return np.asarray(vec, dtype="<f4").tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    """Deserialize a little-endian float32 blob back to a vector."""
    return np.frombuffer(blob, dtype="<f4").copy()


def cosine_search(
    query: np.ndarray,
    matrix: np.ndarray,
    ids: list[str],
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Return the ``top_k`` (id, cosine_similarity) pairs, highest first.

    ``matrix`` is ``(n, dim)`` of already-normalised row vectors aligned with
    ``ids``. ``query`` is normalised here.
    """
    if matrix.size == 0 or len(ids) == 0:
        return []
    q = _l2_normalize(np.asarray(query, dtype=np.float32))
    sims = matrix @ q  # rows are unit vectors -> dot product == cosine
    k = min(top_k, len(ids))
    # argpartition for the top-k, then sort just those.
    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return [(ids[i], float(sims[i])) for i in top_idx]
