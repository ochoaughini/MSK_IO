import numpy as np
from msk_io.image_processing.dl_segmentor import DLSegmentor


def test_dl_segmentor_segment():
    volume = np.random.rand(2, 2, 2).astype(np.float32)
    seg = DLSegmentor()
    mask = seg.segment(volume)
    assert mask.shape == volume.shape
