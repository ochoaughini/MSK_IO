import numpy as np
from msk_io.image_processing.constraint_mapper import ConstraintMapper


def test_map():
    mask = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    mapper = ConstraintMapper({1: 'tissue'})
    res = mapper.map(mask)
    assert res['tissue'] == 3


def test_map_entities():
    mask = np.array([[1, 2], [0, 2]], dtype=np.uint8)
    mapper = ConstraintMapper({1: 'tissue', 2: 'organ'})
    ents = mapper.map_entities(mask)
    stats = {e.name: e.volume for e in ents}
    assert stats['tissue'] == 1
    assert stats['organ'] == 2
