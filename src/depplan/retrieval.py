from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class DenseRetriever:
    def __init__(self, model_path: Path, *, device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError(
                "sentence-transformers could not be imported. Install the package in a compatible "
                "PyTorch environment."
            ) from exc
        self.encoder = SentenceTransformer(str(model_path), device=device)

    def retrieve(
        self,
        embedding_record: dict[str, Any],
        query: str,
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        contexts = embedding_record.get("flat_contexts", [])
        raw_embeddings = embedding_record.get("context_embeddings", [])
        if not contexts or not raw_embeddings:
            raise ValueError("Embedding record has no flat_contexts/context_embeddings")
        matrix = np.asarray(raw_embeddings, dtype=np.float32)
        query_vector = self.encoder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0].astype(np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(contexts):
            raise ValueError("Context text and embedding counts do not match")
        if matrix.shape[1] != query_vector.shape[0]:
            raise ValueError(
                f"Embedding dimension mismatch: contexts={matrix.shape[1]}, query={query_vector.shape[0]}"
            )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-12)
        scores = matrix @ query_vector
        count = min(max(1, top_k), len(contexts))
        indices = np.argpartition(-scores, count - 1)[:count]
        indices = indices[np.argsort(-scores[indices])]
        return [
            {
                "index": int(index),
                "text": str(contexts[index]),
                "score": float(scores[index]),
            }
            for index in indices
        ]
