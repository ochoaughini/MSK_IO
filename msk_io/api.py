from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
from prometheus_client import Counter, Gauge, Histogram

from .decorators import instrument_stage, map_exceptions
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

# Prometheus metrics
VOLUME_COUNTER = Counter("volumes_processed_total", "Volumes processed")
ACTIVE_RUNS = Gauge("active_runs", "Active pipeline runs")
PREDICATE_HIST = Histogram(
    "predicates_per_case", "Number of predicates per case", buckets=(1, 2, 5, 10, 20)
)


@dataclass
class PipelineResult:
    nifti: str
    valid: bool
    hash: str
    predicates: Iterable[str]


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

    def run(self, settings: PipelineSettings, vault: MemoryVault) -> PipelineResult:
        """Run the pipeline synchronously."""
        dicom_dir = settings.data_path
        ACTIVE_RUNS.inc()

        @instrument_stage("load")
        @map_exceptions(DICOMLoadError("", stage="load"))
        def _load() -> np.ndarray:
            return self.loader.load_series(dicom_dir)

        volume = _load()

        @map_exceptions(NiftiConversionError("", stage="conversion"))
        def _convert() -> Path:
            return self.converter.to_nifti(volume, dicom_dir / "volume.nii.gz")

        nifti_path = _convert()

        try:
            self.exporter.save_slice(volume, len(volume) // 2, dicom_dir / "slice.png")
        except Exception:  # pragma: no cover - optional
            pass

        @instrument_stage("segment")
        @map_exceptions(SegmentationError("", stage="segmentation"))
        def _segment() -> np.ndarray:
            return self.segmentor.segment(volume)

        mask = _segment()

        @instrument_stage("map")
        @map_exceptions(MappingError("", stage="mapping"))
        def _map() -> Iterable[str]:
            return self.mapper.map(mask)

        predicates = _map()

        @instrument_stage("emit")
        @map_exceptions(EmissionError("", stage="emission"))
        def _emit() -> SymbolicState:
            return self.emitter.emit_state(np.array([0.5]), np.array([0.5]))

        state = _emit()

        @instrument_stage("validate")
        @map_exceptions(ConstraintValidationError("", stage="validation"))
        def _validate() -> bool:
            lattice = (
                self.lattice or ConstraintLattice(settings.lattice.rules_path)
                if settings.lattice
                else None
            )
            return lattice.validate_chain([state]) if lattice else True

        valid = _validate()

        @instrument_stage("harmonize")
        @map_exceptions(HarmonizationError("", stage="harmonization"))
        def _harmonize() -> SymbolicState:
            return self.harmonizer.harmonize(
                [AgentOutput(state=state, weight=1.0, agent_id="default")]
            )

        final_state = _harmonize()

        @instrument_stage("vault")
        @map_exceptions(VaultError("", stage="vault"))
        def _vault() -> str:
            local_vault = self.vault or vault
            return local_vault.checkpoint(final_state)

        h = _vault()

        VOLUME_COUNTER.inc()
        ACTIVE_RUNS.dec()
        PREDICATE_HIST.observe(len(list(predicates)))
        return PipelineResult(
            nifti=str(nifti_path), valid=valid, hash=h, predicates=predicates
        )

    async def run_async(
        self,
        settings: PipelineSettings,
        vault: MemoryVault,
        callback: Optional[Callable[[PipelineResult], None]] = None,
    ) -> PipelineResult:
        """Asynchronous wrapper for run."""
        import asyncio

        result = await asyncio.to_thread(self.run, settings, vault)
        if callback:
            callback(result)
        return result


def run_pipeline(settings: PipelineSettings, vault: MemoryVault) -> PipelineResult:
    runner = PipelineRunner()
    return runner.run(settings, vault)
