"""Per-question dense, BM25, and reciprocal-rank hybrid retrieval."""
from collections import Counter
import math
import re

import numpy as np


def words(text):
    return re.findall(r"\w+", str(text).lower())


class Encoder:
    def __init__(self, name, device=None):
        self.name, self.device, self.model = name, device, None

    def load(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.name, device=self.device)

    def encode(self, texts):
        self.load()
        return self.model.encode(list(texts), normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)


class PassageIndex:
    def __init__(self, passages, encoder, *, mode='dense', k1=1.2, b=0.75,
                 rrf_k=60):
        self.passages = [dict(p) for p in passages]
        self.encoder = encoder
        self.mode, self.k1, self.b, self.rrf_k = mode, k1, b, rrf_k
        self.texts = [' '.join((p.get('title', ''), p['text'])).strip()
                      for p in self.passages]
        self.counts = [Counter(words(text)) for text in self.texts]
        self.lengths = np.array([sum(c.values()) for c in self.counts], dtype=float)
        self.avg_length = float(self.lengths.mean()) if len(self.lengths) else 1.0
        self.df = Counter(term for c in self.counts for term in c)
        self.matrix = None

    def encode(self, texts):
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        return np.asarray(self.encoder.encode(texts), dtype=np.float32)

    def _bm25(self, query):
        scores = np.zeros(len(self.texts), dtype=float)
        if not self.texts:
            return scores
        norm = self.k1 * (1 - self.b + self.b * self.lengths / max(self.avg_length, 1))
        for token in set(words(query)):
            df = self.df[token]
            idf = math.log(1 + (len(self.texts) - df + 0.5) / (df + 0.5))
            freq = np.array([c[token] for c in self.counts], dtype=float)
            scores += idf * freq * (self.k1 + 1) / np.maximum(freq + norm, 1e-12)
        return scores

    def rank(self, query, k=None, mode=None):
        mode = mode or self.mode
        if mode not in {'dense', 'bm25', 'hybrid'}:
            raise ValueError(f'Unknown retrieval mode: {mode}')
        count = len(self.texts) if k is None else min(len(self.texts), max(0, int(k)))
        if count == 0:
            return []
        sparse = self._bm25(query) if mode in {'bm25', 'hybrid'} else None
        if mode in {'dense', 'hybrid'}:
            if self.matrix is None:
                self.matrix = self.encode(self.texts)
            dense = self.matrix @ self.encode([query])[0]
        if mode == 'dense':
            scores = dense
        elif mode == 'bm25':
            scores = sparse
        else:
            scores = np.zeros(len(self.texts), dtype=float)
            for values in [sparse, dense]:
                order = np.argsort(-values, kind='stable')
                scores[order] += 1 / (self.rrf_k + np.arange(1, len(order) + 1))
        order = np.argsort(-scores, kind='stable')[:count]
        return [dict(self.passages[i], index=int(i), score=float(scores[i])) for i in order]
