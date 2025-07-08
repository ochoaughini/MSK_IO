from __future__ import annotations

from pathlib import Path


class OCRExtractor:
    """Dummy OCR extractor using pytesseract if available."""

    def extract(self, path: Path) -> str:
        try:
            import pytesseract
            from PIL import Image

            return pytesseract.image_to_string(Image.open(path))
        except Exception:
            # Fallback: return file name as text for testing
            return path.stem
