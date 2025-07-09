import numpy as np
import pydicom
from pathlib import Path
from msk_io.preprocessing.dicom_loader import DICOMLoader


def _create_series(path: Path, count: int) -> None:
    for i in range(count):
        arr = (np.random.rand(5, 5) * 255).astype(np.uint16)
        meta = pydicom.dataset.FileMetaDataset()
        meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        ds = pydicom.dataset.FileDataset("test", {}, file_meta=meta, preamble=b"\0" * 128)
        ds.PixelData = arr.tobytes()
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.Rows, ds.Columns = arr.shape
        ds.InstanceNumber = i
        ds.ImagePositionPatient = [0, 0, float(i)]
        ds.save_as(path / f"slice_{i}.dcm")


def test_load_multiple_slices(tmp_path: Path) -> None:
    _create_series(tmp_path, 3)
    loader = DICOMLoader()
    volume = loader.load_series(tmp_path)
    assert volume.shape == (3, 5, 5)
