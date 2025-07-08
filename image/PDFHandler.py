"""Handle PDF loading and page extraction."""
from __future__ import annotations

from pathlib import Path
from typing import List

from pdfminer.high_level import extract_text


class PDFHandler:
    def load(self, path: Path) -> str:
        return extract_text(str(path))

    def split_pages(self, text: str) -> List[str]:
        return text.split("\f")
