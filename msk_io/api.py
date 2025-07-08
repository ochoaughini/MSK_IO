from pathlib import Path
from typing import Optional
import numpy as np
from .preprocessing.dicom_loader import DICOMLoader
from .preprocessing.nifti_converter import NiftiConverter
from .preprocessing.png_exporter import PNGExporter
from .image_processing.segmentor import Segmentor
from .image_processing.constraint_mapper import ConstraintMapper
from .symbolic.symbolic_state_emitter import SymbolicStateEmitter, SymbolicState
from .inference.constraint_lattice import ConstraintLattice
from .control.multi_agent_harmonizer import MultiAgentHarmonizer, AgentOutput
from .storage.memory_vault import MemoryVault
from .config import PipelineSettings
from .errors import (
    DICOMLoadError,
    NiftiConversionError,
    SegmentationError,
    MappingError,
    EmissionError,
    ConstraintValidationError,
    HarmonizationError,
    VaultError,
)


class PipelineRunner:
    def __init__(
        self,
        loader: Optional[DICOMLoader] = None,
        converter: Optional[NiftiConverter] = None,
        exporter: Optional[PNGExporter] = None,
        segmentor: Optional[Segmentor] = None,
        mapper: Optional[ConstraintMapper] = None,
        emitter: Optional[SymbolicStateEmitter] = None,
        lattice: Optional[ConstraintLattice] = None,
        harmonizer: Optional[MultiAgentHarmonizer] = None,
        vault: Optional[MemoryVault] = None,
    ):
        self.loader = loader or DICOMLoader()
        self.converter = converter or NiftiConverter()
        self.exporter = exporter or PNGExporter()
        self.segmentor = segmentor or Segmentor()
        self.mapper = mapper or ConstraintMapper({1: "region"})
        self.emitter = emitter or SymbolicStateEmitter()
        self.lattice = lattice
        self.harmonizer = harmonizer or MultiAgentHarmonizer()
        self.vault = vault

    def run(self, dicom_dir: Path, config: PipelineSettings, vault_path: Path) -> dict:
        try:
            volume = self.loader.load_series(dicom_dir)
        except Exception as exc:
            raise DICOMLoadError(str(exc)) from exc

        try:
            nifti_path = self.converter.to_nifti(volume, dicom_dir / "volume.nii.gz")
        except Exception as exc:
            raise NiftiConversionError(str(exc)) from exc

        try:
            self.exporter.save_slice(volume, len(volume) // 2, dicom_dir / "slice.png")
        except Exception:
            pass

        try:
            mask = self.segmentor.segment(volume)
        except Exception as exc:
            raise SegmentationError(str(exc)) from exc

        try:
            predicates = self.mapper.map(mask)
        except Exception as exc:
            raise MappingError(str(exc)) from exc

        try:
            state = self.emitter.emit_state(np.array([0.5]), np.array([0.5]))
        except Exception as exc:
            raise EmissionError(str(exc)) from exc

        try:
            lattice = self.lattice or ConstraintLattice(config.rules_path)
            valid = lattice.validate_chain([state])
        except Exception as exc:
            raise ConstraintValidationError(str(exc)) from exc

        try:
            final_state = self.harmonizer.harmonize(
                [AgentOutput(state=state, weight=1.0, agent_id="default")]
            )
        except Exception as exc:
            raise HarmonizationError(str(exc)) from exc

        try:
            vault = self.vault or MemoryVault(vault_path)
            h = vault.checkpoint(final_state)
        except Exception as exc:
            raise VaultError(str(exc)) from exc

        return {
            "nifti": str(nifti_path),
            "valid": valid,
            "hash": h,
        }


def run_pipeline(dicom_dir: Path, config: PipelineSettings, vault_path: Path) -> dict:
    runner = PipelineRunner()
    return runner.run(dicom_dir, config, vault_path)
