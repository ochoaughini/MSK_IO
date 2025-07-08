from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class IndexedItem:
    text: str
    embedding: List[float]


def _embed(text: str) -> List[float]:
    return [float(len(text))]


class SemanticIndexer:
    """Trivial semantic indexer using text length as embedding."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.index: Dict[str, IndexedItem] = {}

    def index_items(self, items: List[str]) -> None:
        for text in items:
            self.index[text] = IndexedItem(text=text, embedding=_embed(text))

    def query(self, text: str) -> str:
        if not self.index:
            raise ValueError("Empty index")
        emb = _embed(text)[0]
        closest = min(self.index.values(), key=lambda i: abs(i.embedding[0] - emb))
        return closest.text
