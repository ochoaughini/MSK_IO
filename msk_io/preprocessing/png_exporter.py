from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image

class PNGExporter:
    """Export numpy slices or projections as PNG images."""

    def save_slice(self, volume: np.ndarray, index: int, out_path: Union[str, Path]) -> Path:
        out_path = Path(out_path)
        slice_img = volume[index]
        if slice_img.ndim == 2:
            mode = 'L'
        else:
            mode = 'RGB'
        img = Image.fromarray(slice_img.astype(np.uint8), mode)
        img.save(out_path)
        return out_path
