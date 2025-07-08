from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTImage


@dataclass
class PageImage:
    data: bytes
    bbox: Tuple[float, float, float, float]


@dataclass
class PageItem:
    text: str
    images: List[PageImage]


class PDFLoader:
    """Lightweight PDF loader returning text and images for each page."""

    def load(self, path: Path) -> List[PageItem]:
        if not path.exists():
            raise FileNotFoundError(path)
        pages: List[PageItem] = []
        for page_layout in extract_pages(str(path)):
            text = "".join(
                element.get_text()
                for element in page_layout
                if isinstance(element, LTTextContainer)
            )
            images: List[PageImage] = []
            for element in page_layout:
                if isinstance(element, LTImage) and element.stream:
                    images.append(
                        PageImage(
                            data=bytes(element.stream.get_data()),
                            bbox=element.bbox,
                        )
                    )
            pages.append(PageItem(text=text, images=images))
        return pages
