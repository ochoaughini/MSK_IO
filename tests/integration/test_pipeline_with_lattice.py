import json
import numpy as np
import pydicom
from pathlib import Path
from msk_io.api import PipelineRunner
from msk_io.config import PipelineSettings
from msk_io.storage.memory_vault import MemoryVault


def _create_series(path: Path) -> None:
    arr = np.full((5, 5), 250, dtype=np.uint16)
    meta = pydicom.dataset.FileMetaDataset()
    meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    ds = pydicom.dataset.FileDataset('test', {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PixelData = arr.tobytes()
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.Rows, ds.Columns = arr.shape
    ds.ImagePositionPatient = [0, 0, 0]
    ds.save_as(path / 'slice.dcm')


def test_pipeline_with_lattice(tmp_path: Path) -> None:
    data = tmp_path / 'data'
    data.mkdir()
    _create_series(data)
    rules = tmp_path / 'rules.json'
    rules.write_text(json.dumps({'required_predicates': ['positive'], 'allowed_predicates': ['positive', 'negative']}))
    settings = PipelineSettings(data_path=data, lattice={'rules_path': rules})
    vault = MemoryVault(tmp_path / 'db.sqlite')
    result = PipelineRunner().run(settings, vault)
    assert result.valid
    assert result.predicates

