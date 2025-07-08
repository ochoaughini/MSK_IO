from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Callable, Iterable

import numpy as np
from prometheus_client import Counter, Summary

from .preprocessing.dicom_loader import DICOMLoader
from .preprocessing.nifti_converter import NiftiConverter
from .preprocessing.png_exporter import PNGExporter
from .image_processing.segmentor import Segmentor
from .image_processing.constraint_mapper import ConstraintMapper
from .symbolic.symbolic_state_emitter import SymbolicStateEmitter, SymbolicState
from .inference.constraint_lattice import ConstraintLattice
from .control.multi_agent_harmonizer import MultiAgentHarmonizer, AgentOutput
from .storage.memory_vault import MemoryVault
from .config import PipelineConfig
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

# Basic Prometheus metrics
LOAD_TIME = Summary("dicom_load_seconds", "Time spent loading DICOM")
SEG_TIME = Summary("segment_seconds", "Time spent segmenting")
VOLUME_COUNTER = Counter("volumes_processed_total", "Volumes processed")


class PipelineRunner:
    """Encapsulates the pipeline execution."""

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
    ) -> None:
        self.loader = loader or DICOMLoader()
        self.converter = converter or NiftiConverter()
        self.exporter = exporter or PNGExporter()
        self.segmentor = segmentor or Segmentor()
        self.mapper = mapper or ConstraintMapper({1: "region"})
        self.emitter = emitter or SymbolicStateEmitter()
        self.lattice = lattice
        self.harmonizer = harmonizer or MultiAgentHarmonizer()
        self.vault = vault

    def run(self, dicom_dir: Path, config: PipelineConfig, vault_path: Path) -> dict:
        """Run the pipeline synchronously."""
        with LOAD_TIME.time():
            try:
                volume = self.loader.load_series(dicom_dir)
            except Exception as exc:  # pragma: no cover - wrapped
                raise DICOMLoadError(str(exc)) from exc

        try:
            nifti_path = self.converter.to_nifti(volume, dicom_dir / "volume.nii.gz")
        except Exception as exc:  # pragma: no cover - wrapped
            raise NiftiConversionError(str(exc)) from exc

        try:
            self.exporter.save_slice(volume, len(volume) // 2, dicom_dir / "slice.png")
        except Exception:  # pragma: no cover - optional
            pass

        with SEG_TIME.time():
            try:
                mask = self.segmentor.segment(volume)
            except Exception as exc:  # pragma: no cover - wrapped
                raise SegmentationError(str(exc)) from exc

        try:
            predicates = self.mapper.map(mask)
        except Exception as exc:  # pragma: no cover - wrapped
            raise MappingError(str(exc)) from exc

        try:
            state = self.emitter.emit_state(np.array([0.5]), np.array([0.5]))
        except Exception as exc:  # pragma: no cover - wrapped
            raise EmissionError(str(exc)) from exc

        try:
            lattice = (
                self.lattice or ConstraintLattice(config.lattice.rules_path)
                if config.lattice
                else None
            )
            valid = lattice.validate_chain([state]) if lattice else True
        except Exception as exc:  # pragma: no cover - wrapped
            raise ConstraintValidationError(str(exc)) from exc

        try:
            final_state = self.harmonizer.harmonize(
                [AgentOutput(state=state, weight=1.0, agent_id="default")]
            )
        except Exception as exc:  # pragma: no cover - wrapped
            raise HarmonizationError(str(exc)) from exc

        try:
            vault = self.vault or MemoryVault(vault_path)
            h = vault.checkpoint(final_state)
        except Exception as exc:  # pragma: no cover - wrapped
            raise VaultError(str(exc)) from exc

        VOLUME_COUNTER.inc()
        return {
            "nifti": str(nifti_path),
            "valid": valid,
            "hash": h,
            "predicates": predicates,
        }

    async def run_async(
        self,
        dicom_dir: Path,
        config: PipelineConfig,
        vault_path: Path,
        callback: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        """Asynchronous wrapper for run."""
        result = self.run(dicom_dir, config, vault_path)
        if callback:
            callback(result)
        return result


def run_pipeline(dicom_dir: Path, config: PipelineConfig, vault_path: Path) -> dict:
    runner = PipelineRunner()
    return runner.run(dicom_dir, config, vault_path)
