from __future__ import annotations

from pathlib import Path
from typing import List

from .pdf_loader import PDFLoader
from ..ocr.ocr_extractor import OCRExtractor


class MSKPDFIngestor:
    def __init__(self, loader: PDFLoader | None = None, ocr: OCRExtractor | None = None) -> None:
        self.loader = loader or PDFLoader()
        self.ocr = ocr or OCRExtractor()

    def ingest(self, path: Path) -> List[str]:
        pages = self.loader.load(path)
        texts = [p.text for p in pages]
        return texts
