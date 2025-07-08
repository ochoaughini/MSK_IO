"""Placeholder alignment utilities for TotalSegmentatorV2."""
from __future__ import annotations

import numpy as np


def rigid_register(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return source volume aligned to target using a dummy rigid transform."""
    # In a real implementation this would compute a rigid registration.
    return source


def affine_register(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return source volume aligned to target using an affine transform."""
    return source
