from __future__ import annotations

from pathlib import Path
from typing import Union, List, Tuple
import numpy as np
import pydicom
import lmdb
import io
import asyncio
import logging


class InvalidDICOMError(Exception):
    """Raised when a file cannot be parsed as DICOM."""


class DICOMLoader:
    """Load a series of DICOM slices into a 3D numpy array."""

    def __init__(self, cache_lmdb: Path | None = None) -> None:
        self.cache_lmdb = cache_lmdb
        self._env = (
            lmdb.open(str(cache_lmdb), map_size=1_000_000_000)
            if cache_lmdb
            else None
        )
        self.logger = logging.getLogger(__name__)

    def _read_file(self, fp: Path) -> Tuple[float, np.ndarray] | None:
        key = str(fp).encode()
        if self._env:
            with self._env.begin() as txn:
                data = txn.get(key)
                if data:
                    arr = np.load(io.BytesIO(data))
                    return self._slice_key(fp), arr
        try:
            ds = pydicom.dcmread(str(fp))
            arr = ds.pixel_array
            if self._env:
                buf = io.BytesIO()
                np.save(buf, arr)
                with self._env.begin(write=True) as txn:
                    txn.put(key, buf.getvalue())
            return self._slice_key(fp), arr
        except Exception as exc:
            self.logger.warning("Failed to read %s: %s", fp, exc)
            return None

    def load_series(self, path: Union[str, Path]) -> np.ndarray:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        files: List[Path] = sorted(path.glob("*.dcm"))
        if not files:
            raise InvalidDICOMError("No DICOM files found")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = [loop.run_in_executor(None, self._read_file, f) for f in files]
        results = [r for r in loop.run_until_complete(asyncio.gather(*tasks)) if r]
        loop.close()
        if not results:
            raise InvalidDICOMError("All DICOM reads failed")
        results.sort(key=lambda x: x[0])
        ipps = [r[0] for r in results]
        if len(ipps) > 1:
            spacings = np.diff(sorted(ipps))
            if np.std(spacings) > 1e-3:
                self.logger.warning("Inconsistent slice spacing detected")
        volume = np.stack([r[1] for r in results])
        return volume

    def _slice_key(self, fp: Path) -> float:
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True)
            ipp = ds.get("ImagePositionPatient")
            if ipp:
                return float(ipp[2])
            return float(ds.get("InstanceNumber", 0))
        except Exception:
            return 0.0

    async def load_series_async(self, path: Union[str, Path]) -> np.ndarray:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load_series, path)
