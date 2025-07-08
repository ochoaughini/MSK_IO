from __future__ import annotations

from typing import Dict, List

from ..indexer.semantic_indexer import SemanticIndexer


class ConstraintRetriever:
    def __init__(self, indexer: SemanticIndexer) -> None:
        self.indexer = indexer

    def retrieve(self, query: str) -> Dict[str, str]:
        text = self.indexer.query(query)
        return {"text": text, "meta": "example"}
