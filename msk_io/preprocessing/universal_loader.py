from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
import nibabel as nib

from .dicom_loader import DICOMLoader


class UniversalLoader:
    """Load medical images from various formats."""

    def __init__(self, dicom_loader: DICOMLoader | None = None) -> None:
        self.dicom_loader = dicom_loader or DICOMLoader()

    def load(self, path: Union[str, Path]) -> np.ndarray:
        """Load an image volume or array from ``path``.

        * Directories are treated as DICOM series.
        * ``.nii`` or ``.nii.gz`` files are loaded via ``nibabel``.
        * ``.png``/``.jpg`` files return a 2D array.
        """
        p = Path(path)
        if p.is_dir():
            return self.dicom_loader.load_series(p)
        suffix = p.suffix.lower()
        if suffix == ".nii" or str(p).lower().endswith(".nii.gz"):
            nifti = nib.load(str(p))
            data = nifti.get_fdata()
            return np.asarray(data)
        if suffix in {".png", ".jpg", ".jpeg"}:
            img = Image.open(p)
            return np.array(img)
        raise ValueError(f"Unsupported format: {p}")
