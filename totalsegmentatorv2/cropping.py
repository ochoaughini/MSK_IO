"""Simple volume cropping utilities."""
from __future__ import annotations

import numpy as np


def crop_to_mask(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Crop ``volume`` to the bounding box of ``mask``."""
    if volume.shape != mask.shape:
        raise ValueError("volume and mask shapes must match")
    coords = np.argwhere(mask)
    if coords.size == 0:
        return volume
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    slices = tuple(slice(int(mn), int(mx)) for mn, mx in zip(mins, maxs))
    return volume[slices]
