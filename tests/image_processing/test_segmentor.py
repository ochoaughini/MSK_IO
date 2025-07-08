import numpy as np
from msk_io.image_processing.segmentor import Segmentor


def test_segment():
    vol = np.array([[[0, 1], [2, 3]]], dtype=np.uint8)
    seg = Segmentor(threshold=0.5)
    mask = seg.segment(vol)
    assert mask.shape == vol.shape
    assert set(np.unique(mask)) <= {0, 1}
