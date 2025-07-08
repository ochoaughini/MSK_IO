from pathlib import Path
from PIL import Image, ImageDraw

from msk_io.pdf.pdf_loader import PDFLoader


def _create_pdf(path: Path) -> Path:
    img = Image.new("RGB", (200, 200), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "Figure 1: labral tear", fill=(0, 0, 0))
    img.save(path, "PDF")
    return path


def test_pdf_loader(tmp_path: Path) -> None:
    pdf = _create_pdf(tmp_path / "MSK.pdf")
    loader = PDFLoader()
    pages = loader.load(pdf)
    assert len(pages) == 1
    # text may be empty since PIL embeds image-only PDF
    assert isinstance(pages[0].text, str)
