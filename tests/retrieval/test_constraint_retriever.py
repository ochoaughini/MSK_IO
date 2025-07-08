from pathlib import Path
from msk_io.indexer.semantic_indexer import SemanticIndexer
from msk_io.retrieval.constraint_retriever import ConstraintRetriever


def test_constraint_retriever(tmp_path: Path) -> None:
    idx = SemanticIndexer(tmp_path / "index")
    idx.index_items(["Figure: labral tear"])
    retr = ConstraintRetriever(idx)
    res = retr.retrieve("labral tear")
    assert "text" in res
