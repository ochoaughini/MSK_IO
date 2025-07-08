from pathlib import Path
from msk_io.nlp.text_extractor import TextExtractor


def test_extract_text(tmp_path: Path):
    file = tmp_path / 'a.txt'
    file.write_text('hello')
    extractor = TextExtractor()
    assert extractor.extract_text(file) == 'hello'
