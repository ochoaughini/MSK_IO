from pathlib import Path
from typing import Union
import numpy as np
import nibabel as nib

class NiftiConverter:
    """Convert numpy volumes to NIfTI format."""

    def to_nifti(self, volume: np.ndarray, out_path: Union[str, Path]) -> Path:
        out_path = Path(out_path)
        nifti = nib.Nifti1Image(volume, affine=np.eye(4))
        nib.save(nifti, str(out_path))
        return out_path
