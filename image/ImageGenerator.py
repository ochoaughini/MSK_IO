"""Placeholder image generator."""
from __future__ import annotations

from PIL import Image
import numpy as np


class ImageGenerator:
    def generate(self, size: int = 64) -> Image.Image:
        arr = (np.random.rand(size, size) * 255).astype("uint8")
        return Image.fromarray(arr)
