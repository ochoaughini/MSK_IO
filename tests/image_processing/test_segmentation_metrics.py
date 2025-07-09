import numpy as np
from msk_io.image_processing.segmentor import Segmentor


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    return 2 * inter / float(a.sum() + b.sum()) if (a.sum() + b.sum()) else 1.0


def test_segmentation_dice() -> None:
    vol = np.array([[[0, 1], [1, 0]]], dtype=np.uint8)
    seg = Segmentor(threshold=0.5)
    mask = seg.segment(vol)
    assert _dice(mask, vol) == 1.0
