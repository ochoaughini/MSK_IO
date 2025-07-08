from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineError(Exception):
    """Base class for pipeline errors with codes and hints."""

    message: str
    code: str = "PIPELINE_ERROR"
    severity: str = "error"
    hint: str | None = None

    def __str__(self) -> str:  # pragma: no cover - string repr
        return f"{self.code}: {self.message}" + (
            f" Hint: {self.hint}" if self.hint else ""
        )


class DICOMLoadError(PipelineError):
    code = "DICOM_LOAD_FAILED"
    hint = "Check input directory for valid DICOM files."


class NiftiConversionError(PipelineError):
    code = "NIFTI_CONVERSION_FAILED"


class SegmentationError(PipelineError):
    code = "SEGMENTATION_FAILED"


class MappingError(PipelineError):
    code = "MAPPING_FAILED"


class EmissionError(PipelineError):
    code = "EMISSION_FAILED"


class ConstraintValidationError(PipelineError):
    code = "LATTICE_VALIDATION_FAILED"


class HarmonizationError(PipelineError):
    code = "HARMONIZATION_FAILED"


class VaultError(PipelineError):
    code = "VAULT_FAILED"
