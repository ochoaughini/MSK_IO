from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Iterable, Optional

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
from .pdf.pdf_loader import PDFLoader
from .ocr.ocr_extractor import OCRExtractor
from .indexer.semantic_indexer import SemanticIndexer
from .retrieval.constraint_retriever import ConstraintRetriever
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
        pdf_loader: Optional[PDFLoader] = None,
        ocr: Optional[OCRExtractor] = None,
        indexer: Optional[SemanticIndexer] = None,
        retriever: Optional[ConstraintRetriever] = None,
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
        self.pdf_loader = pdf_loader or PDFLoader()
        self.ocr = ocr or OCRExtractor()
        self.indexer = indexer or SemanticIndexer(Path("index"))
        self.retriever = retriever or ConstraintRetriever(self.indexer)

    def run(self, settings: PipelineSettings, vault: MemoryVault) -> dict:
        """Run the pipeline synchronously."""
        dicom_dir = settings.data_path
        with LOAD_TIME.time():
            try:
                volume = self.loader.load_series(dicom_dir)
            except Exception as exc:  # pragma: no cover - wrapped
                raise DICOMLoadError(str(exc), stage="load") from exc

        try:
            nifti_path = self.converter.to_nifti(volume, dicom_dir / "volume.nii.gz")
        except Exception as exc:  # pragma: no cover - wrapped
            raise NiftiConversionError(str(exc), stage="conversion") from exc

        try:
            self.exporter.save_slice(volume, len(volume) // 2, dicom_dir / "slice.png")
        except Exception:  # pragma: no cover - optional
            pass

        with SEG_TIME.time():
            try:
                mask = self.segmentor.segment(volume)
            except Exception as exc:  # pragma: no cover - wrapped
                raise SegmentationError(str(exc), stage="segmentation") from exc

        try:
            predicates = self.mapper.map(mask)
        except Exception as exc:  # pragma: no cover - wrapped
            raise MappingError(str(exc), stage="mapping") from exc

        try:
            state = self.emitter.emit_state(np.array([0.5]), np.array([0.5]))
        except Exception as exc:  # pragma: no cover - wrapped
            raise EmissionError(str(exc), stage="emission") from exc

        try:
            lattice = (
                self.lattice or ConstraintLattice(settings.lattice.rules_path)
                if settings.lattice
                else None
            )
            valid = lattice.validate_chain([state]) if lattice else True
        except Exception as exc:  # pragma: no cover - wrapped
            raise ConstraintValidationError(str(exc), stage="validation") from exc

        try:
            final_state = self.harmonizer.harmonize(
                [AgentOutput(state=state, weight=1.0, agent_id="default")]
            )
        except Exception as exc:  # pragma: no cover - wrapped
            raise HarmonizationError(str(exc), stage="harmonization") from exc

        try:
            vault = self.vault or vault
            h = vault.checkpoint(final_state)
        except Exception as exc:  # pragma: no cover - wrapped
            raise VaultError(str(exc), stage="vault") from exc

        VOLUME_COUNTER.inc()
        return {
            "nifti": str(nifti_path),
            "valid": valid,
            "hash": h,
            "predicates": predicates,
        }

    async def run_async(
        self,
        settings: PipelineSettings,
        vault: MemoryVault,
        callback: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        """Asynchronous wrapper for run."""
        result = self.run(settings, vault)
        if callback:
            callback(result)
        return result


def run_pipeline(settings: PipelineSettings, vault: MemoryVault) -> dict:
    runner = PipelineRunner()
    return runner.run(settings, vault)
