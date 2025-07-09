from pathlib import Path
from typing import Union
import warnings
import numpy as np
import nibabel as nib

class NiftiConverter:
    """Convert numpy volumes to NIfTI format."""

    def to_nifti(self, volume: np.ndarray, out_path: Union[str, Path]) -> Path:
        out_path = Path(out_path)
        try:
            nifti = nib.Nifti1Image(volume, affine=np.eye(4))
        except Exception as exc:
            warnings.warn(f"Conversion failed ({exc}); using fallback", RuntimeWarning)
            arr = np.asarray(volume, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[..., None]
            nifti = nib.Nifti1Image(arr, affine=np.zeros((4, 4)))
        nib.save(nifti, str(out_path))
        return out_path
