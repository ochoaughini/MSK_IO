from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Callable
import asyncio
import numpy as np


def process_slices(volume: np.ndarray, func: Callable[[np.ndarray], np.ndarray], *, workers: int | None = None, backend: str = "thread") -> list[np.ndarray]:
    """Apply ``func`` to each slice of ``volume`` concurrently."""
    executor_cls = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
    with executor_cls(max_workers=workers) as exe:
        futures = [exe.submit(func, volume[i]) for i in range(volume.shape[0])]
        return [f.result() for f in futures]


async def process_slices_async(volume: np.ndarray, func: Callable[[np.ndarray], np.ndarray], *, workers: int | None = None, backend: str = "thread") -> list[np.ndarray]:
    """Asynchronous variant of :func:`process_slices`."""
    loop = asyncio.get_running_loop()
    executor_cls = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
    with executor_cls(max_workers=workers) as exe:
        tasks = [loop.run_in_executor(exe, func, volume[i]) for i in range(volume.shape[0])]
        return await asyncio.gather(*tasks)
