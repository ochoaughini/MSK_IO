import os
import pytest
from msk_io.indexer.semantic_indexer import SemanticIndexer


@pytest.fixture
def indexer_instance(test_config):
    return SemanticIndexer(test_config)


def test_index_document_creates_file(indexer_instance, tmp_path):
    doc_id = 'test_doc'
    text = 'some text'
    indexer_instance.vector_db_path = str(tmp_path / 'db')
    indexer_instance.index_document(doc_id, text, {'meta': 'data'})
    out_file = f'{indexer_instance.vector_db_path}_{doc_id}.txt'
    assert os.path.exists(out_file)


def test_query_semantic_index_returns_results(indexer_instance):
    results = indexer_instance.query_semantic_index('test query', top_k=1)
    assert isinstance(results, list)
    assert len(results) == 1
