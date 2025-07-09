import numpy as np
import pydicom
from pathlib import Path
from PIL import Image
import nibabel as nib

from msk_io.preprocessing.universal_loader import UniversalLoader


def _create_dicom(path: Path) -> Path:
    arr = (np.random.rand(8, 8) * 255).astype(np.uint16)
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    ds = pydicom.dataset.FileDataset("test", {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PixelData = arr.tobytes()
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.Rows, ds.Columns = arr.shape
    ds.ImagePositionPatient = [0, 0, 0]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.save_as(path / "slice.dcm")
    return path


def _create_nifti(path: Path) -> Path:
    data = np.random.rand(2, 2, 2)
    nifti = nib.Nifti1Image(data, affine=np.eye(4))
    out = path / "volume.nii.gz"
    nib.save(nifti, str(out))
    return out


def _create_png(path: Path) -> Path:
    img = Image.new("L", (4, 4), color=0)
    out = path / "img.png"
    img.save(out)
    return out


def test_load_dicom(tmp_path: Path) -> None:
    dicom_dir = _create_dicom(tmp_path)
    loader = UniversalLoader()
    vol = loader.load(dicom_dir)
    assert vol.shape == (1, 8, 8)


def test_load_nifti(tmp_path: Path) -> None:
    nifti = _create_nifti(tmp_path)
    loader = UniversalLoader()
    vol = loader.load(nifti)
    assert vol.shape == (2, 2, 2)


def test_load_png(tmp_path: Path) -> None:
    img = _create_png(tmp_path)
    loader = UniversalLoader()
    arr = loader.load(img)
    assert arr.shape == (4, 4)
