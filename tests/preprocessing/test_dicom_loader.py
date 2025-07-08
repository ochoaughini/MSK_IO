import numpy as np
import pydicom
from pathlib import Path
from msk_io.preprocessing.dicom_loader import DICOMLoader


def test_load_series(tmp_path: Path):
    arr = (np.random.rand(10, 10) * 255).astype(np.uint16)
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    ds = pydicom.dataset.FileDataset('test', {}, file_meta=file_meta, preamble=b"\0" * 128)
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
    ds.save_as(tmp_path / 'slice.dcm')

    loader = DICOMLoader()
    volume = loader.load_series(tmp_path)
    assert volume.shape == (1, 10, 10)


def test_slice_orientation(tmp_path: Path) -> None:
    arr = (np.random.rand(5, 5) * 255).astype(np.uint16)
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    ds = pydicom.dataset.FileDataset('test', {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PixelData = arr.tobytes()
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.Rows, ds.Columns = arr.shape
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    p = tmp_path / 'slice.dcm'
    ds.save_as(p)
    loader = DICOMLoader()
    orient = loader._slice_orientation(p)
    assert orient == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
