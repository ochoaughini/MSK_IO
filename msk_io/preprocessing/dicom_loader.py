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
        files: List[Path] = sorted(path.glob('*.dcm'))
        if not files:
            raise InvalidDICOMError('No DICOM files found')
        slices = [self._read_file(f) for f in files]
        return np.stack(slices)
