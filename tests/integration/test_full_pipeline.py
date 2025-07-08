import numpy as np
import pydicom
from pathlib import Path
from msk_io.api import PipelineRunner, PipelineResult
from msk_io.config import PipelineSettings
from msk_io.storage.memory_vault import MemoryVault


def create_dicom_series(dir: Path):
    arr = (np.random.rand(10, 10) * 255).astype(np.uint16)
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
    ds.save_as(dir / "slice.dcm")


def test_full_pipeline(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    create_dicom_series(data)
    settings = PipelineSettings(data_path=data)
    vault = MemoryVault(tmp_path / "vault.db")
    result = PipelineRunner().run(settings, vault)
    assert isinstance(result, PipelineResult)
    assert Path(result.nifti).exists()
    assert result.valid is True
    assert isinstance(result.entities, list)
