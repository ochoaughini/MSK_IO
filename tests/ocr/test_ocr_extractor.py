from pathlib import Path
from PIL import Image, ImageDraw

from msk_io.ocr.ocr_extractor import OCRExtractor


def _create_img(path: Path) -> Path:
    img = Image.new("RGB", (100, 50), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((5, 5), "shoulder", fill=(0, 0, 0))
    img.save(path)
    return path


def test_ocr_extractor(tmp_path: Path) -> None:
    img_path = _create_img(tmp_path / "shoulder.png")
    extractor = OCRExtractor()
    text = extractor.extract(img_path)
    assert "shoulder" in text
