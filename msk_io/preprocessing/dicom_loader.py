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
    """Load a series of DICOM slices into a 3D numpy array.

    The loader attempts to verify basic alignment by ensuring that slice
    orientations are consistent across the series. Any detected mismatch is
    logged as a warning but does not stop loading.
    """

    def __init__(self, cache_lmdb: Path | None = None) -> None:
        self.cache_lmdb = cache_lmdb
        self._env = (
            lmdb.open(str(cache_lmdb), map_size=1_000_000_000)
            if cache_lmdb
            else None
        )
        self.logger = logging.getLogger(__name__)

    def _slice_orientation(self, fp: Path) -> Tuple[float, ...] | None:
        """Return the ImageOrientationPatient tuple if present."""
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True)
            iop = ds.get("ImageOrientationPatient")
            if iop:
                return tuple(float(x) for x in iop)
        except Exception:
            pass
        return None

    def _check_orientation(self, paths: List[Path]) -> None:
        """Log a warning if orientations differ significantly."""
        orientations = [self._slice_orientation(p) for p in paths]
        valid = [o for o in orientations if o]
        if len(valid) <= 1:
            return
        base = np.array(valid[0])
        for orient in valid[1:]:
            diff = np.linalg.norm(np.array(orient) - base)
            if diff > 1e-3:
                self.logger.warning("Inconsistent slice orientation detected")
                break

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
        self._check_orientation(files)
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
