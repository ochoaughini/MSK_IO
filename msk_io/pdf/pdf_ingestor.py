from __future__ import annotations

from pathlib import Path
from typing import List

from .pdf_loader import PDFLoader
from ..ocr.ocr_extractor import OCRExtractor


class MSKPDFIngestor:
    def __init__(self, loader: PDFLoader | None = None, ocr: OCRExtractor | None = None) -> None:
        self.loader = loader or PDFLoader()
        self.ocr = ocr or OCRExtractor()

    def ingest(self, path: Path, ocr_enabled: bool = False) -> List[str]:
        pages = self.loader.load(path)
        texts: List[str] = []
        for page in pages:
            text = page.text.strip()
            if ocr_enabled and (not text or len(text) < 5) and page.images:
                ocr_texts = [self.ocr.extract_bytes(img.data) for img in page.images]
                if ocr_texts:
                    text = " ".join(ocr_texts)
            texts.append(text)
        return texts
