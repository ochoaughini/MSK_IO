from __future__ import annotations

from typing import List

from ..indexer.semantic_indexer import SemanticIndexer


class ConstraintRetriever:
    def __init__(self, indexer: SemanticIndexer) -> None:
        self.indexer = indexer

    def retrieve(self, query: str) -> str:
        return self.indexer.query(query)
