import asyncio
import numpy as np
from msk_io.image_processing.parallel_executor import (
    process_slices,
    process_slices_async,
)


def _sum(slice_: np.ndarray) -> np.ndarray:
    return np.array([slice_.sum()])


def test_process_slices() -> None:
    vol = np.arange(12).reshape(3, 2, 2)
    res = process_slices(vol, _sum)
    assert len(res) == 3
    assert res[0] == np.array([0 + 1 + 2 + 3])[:1]


def test_process_slices_async() -> None:
    vol = np.arange(8).reshape(2, 2, 2)
    res = asyncio.run(process_slices_async(vol, _sum))
    assert len(res) == 2
    assert res[1] == np.array([4 + 5 + 6 + 7])[:1]
