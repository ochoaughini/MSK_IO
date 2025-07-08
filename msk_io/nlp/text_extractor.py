from pathlib import Path
from typing import Union

class TextExtractor:
    """Extract text from documents via OCR or PDF parsing."""

    def extract_text(self, path: Union[str, Path]) -> str:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() == '.txt':
            return path.read_text()
        try:
            import pdfminer.high_level as pdfminer
            if path.suffix.lower() == '.pdf':
                return pdfminer.extract_text(str(path))
        except Exception:
            pass
        try:
            import pytesseract
            from PIL import Image
            if path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                return pytesseract.image_to_string(Image.open(path))
        except Exception:
            pass
        raise ValueError('Unsupported file type or missing dependencies')
