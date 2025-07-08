from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer


@dataclass
class PDFPage:
    text: str


class PDFLoader:
    """Lightweight PDF loader returning text for each page."""

    def load(self, path: Path) -> List[PDFPage]:
        if not path.exists():
            raise FileNotFoundError(path)
        pages = []
        for page_layout in extract_pages(str(path)):
            text = "".join(
                element.get_text() for element in page_layout if isinstance(element, LTTextContainer)
            )
            pages.append(PDFPage(text=text))
        return pages
