import numpy as np
from msk_io.image_processing.constraint_mapper import ConstraintMapper


def test_map():
    mask = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    mapper = ConstraintMapper({1: 'tissue'})
    res = mapper.map(mask)
    assert res['tissue'] == 3
