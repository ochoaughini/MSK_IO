from __future__ import annotations

from pathlib import Path
from typing import Union, List
import numpy as np
import pydicom


class InvalidDICOMError(Exception):
    """Raised when a file cannot be parsed as DICOM."""


class DICOMLoader:
    """Load a series of DICOM slices into a 3D numpy array."""

    def _read_file(self, fp: Path) -> np.ndarray:
        try:
            ds = pydicom.dcmread(str(fp))
            return ds.pixel_array
        except Exception as exc:
            raise InvalidDICOMError(str(exc)) from exc

    def load_series(self, path: Union[str, Path]) -> np.ndarray:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        files: List[Path] = sorted(path.glob("*.dcm"))
        if not files:
            raise InvalidDICOMError("No DICOM files found")
        slices = []
        for f in files:
            arr = self._read_file(f)
            slices.append((self._slice_key(f), arr))
        slices.sort(key=lambda x: x[0])
        volume = np.stack([s[1] for s in slices])
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
        return self.load_series(path)
