"""Utilities for loading and saving images."""
from __future__ import annotations

from pathlib import Path
from PIL import Image


class ImageHandler:
    def load(self, path: Path) -> Image.Image:
        return Image.open(path)

    def save(self, img: Image.Image, path: Path) -> None:
        img.save(path)
