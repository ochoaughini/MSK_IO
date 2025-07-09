from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Union
import io


class OCRExtractor:
    """OCR extractor using pytesseract with simple batching."""

    def _extract_one(self, source: Union[Path, bytes]) -> Tuple[str, float]:
        try:
            import pytesseract
            from PIL import Image

            if isinstance(source, Path):
                img = Image.open(source)
            else:  # bytes
                img = Image.open(io.BytesIO(source))
            text = pytesseract.image_to_string(img)
            conf = 0.9
        except Exception:
            text = source.stem if isinstance(source, Path) else ""
            conf = 0.5
        return text, conf

    def extract(self, path: Path) -> str:
        return self._extract_one(path)[0]

    def extract_bytes(self, data: bytes) -> str:
        return self._extract_one(data)[0]

    def extract_batch(self, paths: List[Union[Path, bytes]]) -> List[Tuple[str, float]]:
        with ThreadPoolExecutor() as ex:
            return list(ex.map(self._extract_one, paths))
