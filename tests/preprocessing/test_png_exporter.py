import numpy as np
from pathlib import Path
from PIL import Image
from msk_io.preprocessing.png_exporter import PNGExporter


def test_save_slice(tmp_path: Path):
    vol = (np.random.rand(1, 10, 10) * 255).astype(np.uint8)
    exporter = PNGExporter()
    out = exporter.save_slice(vol, 0, tmp_path / 'slice.png')
    assert out.exists()
    img = Image.open(out)
    assert img.size == (10, 10)
