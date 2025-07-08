from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
try:
    import faiss
except Exception:  # pragma: no cover - optional
    faiss = None


@dataclass
class IndexedItem:
    text: str
    embedding: List[float]


def _embed(text: str) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.random(16).astype("float32")


class SemanticIndexer:
    """FAISS-backed semantic indexer with random embeddings."""

    def __init__(self, path: Path) -> None:
        self.path = path
        if faiss:
            self.index = faiss.IndexFlatL2(16)
        else:  # pragma: no cover - fallback
            self.index = None
        self.embeddings: List[np.ndarray] = []
        self.items: List[str] = []

    def index_items(self, items: List[str]) -> None:
        embeddings = np.vstack([_embed(t) for t in items])
        if self.index:
            self.index.add(embeddings)
        else:
            self.embeddings.extend(list(embeddings))
        self.items.extend(items)

    def add_items(self, items: List[str]) -> None:
        self.index_items(items)

    def query(self, text: str) -> str:
        if not self.items:
            raise ValueError("Empty index")
        emb = _embed(text).reshape(1, -1)
        if self.index:
            _, idx = self.index.search(emb, 1)
            return self.items[int(idx[0][0])]
        else:
            dists = [float(np.linalg.norm(e - emb)) for e in self.embeddings]
            i = int(np.argmin(dists))
            return self.items[i]

    def query_batch(self, texts: List[str]) -> List[str]:
        if not self.items:
            raise ValueError("Empty index")
        embs = np.vstack([_embed(t) for t in texts])
        if self.index:
            _, idx = self.index.search(embs, 1)
            return [self.items[int(i[0])] for i in idx]
        else:
            results = []
            for emb in embs:
                dists = [float(np.linalg.norm(e - emb)) for e in self.embeddings]
                i = int(np.argmin(dists))
                results.append(self.items[i])
            return results
