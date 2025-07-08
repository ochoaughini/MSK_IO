from pathlib import Path
import numpy as np
from .preprocessing.dicom_loader import DICOMLoader
from .preprocessing.nifti_converter import NiftiConverter
from .preprocessing.png_exporter import PNGExporter
from .image_processing.segmentor import Segmentor
from .image_processing.constraint_mapper import ConstraintMapper
from .symbolic.symbolic_state_emitter import SymbolicStateEmitter
from .inference.constraint_lattice import ConstraintLattice
from .control.multi_agent_harmonizer import MultiAgentHarmonizer, AgentOutput
from .storage.memory_vault import MemoryVault


def run_pipeline(dicom_dir: Path, config_path: Path, vault_path: Path) -> dict:
    loader = DICOMLoader()
    volume = loader.load_series(dicom_dir)

    converter = NiftiConverter()
    nifti_path = converter.to_nifti(volume, dicom_dir / 'volume.nii.gz')

    exporter = PNGExporter()
    exporter.save_slice(volume, len(volume)//2, dicom_dir / 'slice.png')

    segmentor = Segmentor()
    mask = segmentor.segment(volume)

    mapper = ConstraintMapper({1: 'region'})
    predicates = mapper.map(mask)

    emitter = SymbolicStateEmitter()
    state = emitter.emit_state(np.array([0.5]), np.array([0.5]))

    lattice = ConstraintLattice(config_path)
    valid = lattice.validate_chain([state])

    harmonizer = MultiAgentHarmonizer()
    final_state = harmonizer.harmonize([AgentOutput(state, 1.0)])

    vault = MemoryVault(vault_path)
    h = vault.checkpoint(final_state)

    return {
        'nifti': str(nifti_path),
        'valid': valid,
        'hash': h,
    }
