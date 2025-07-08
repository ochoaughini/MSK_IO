import numpy as np
from pathlib import Path
import nibabel as nib
from msk_io.preprocessing.nifti_converter import NiftiConverter


def test_to_nifti(tmp_path: Path):
    vol = np.zeros((2, 2, 2), dtype=np.float32)
    converter = NiftiConverter()
    out = converter.to_nifti(vol, tmp_path / 'test.nii.gz')
    assert out.exists()
    loaded = nib.load(str(out)).get_fdata()
    assert loaded.shape == vol.shape
