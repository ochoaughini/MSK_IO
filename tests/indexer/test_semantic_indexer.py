from pathlib import Path
from msk_io.indexer.semantic_indexer import SemanticIndexer


def test_semantic_indexer(tmp_path: Path) -> None:
    index_path = tmp_path / "index"
    idx = SemanticIndexer(index_path)
    idx.index_items(["labral tear", "meniscus injury"])
    result = idx.query("labral damage")
    assert result == "labral tear"
