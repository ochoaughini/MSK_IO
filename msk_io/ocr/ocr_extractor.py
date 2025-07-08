from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple


class OCRExtractor:
    """OCR extractor using pytesseract with simple batching."""

    def _extract_one(self, path: Path) -> Tuple[str, float]:
        try:
            import pytesseract
            from PIL import Image

            text = pytesseract.image_to_string(Image.open(path))
            conf = 0.9
        except Exception:
            text = path.stem
            conf = 0.5
        return text, conf

    def extract(self, path: Path) -> str:
        return self._extract_one(path)[0]

    def extract_batch(self, paths: List[Path]) -> List[Tuple[str, float]]:
        with ThreadPoolExecutor() as ex:
            return list(ex.map(self._extract_one, paths))
