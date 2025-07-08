class PipelineError(Exception):
    """Base class for pipeline errors."""


class DICOMLoadError(PipelineError):
    pass


class NiftiConversionError(PipelineError):
    pass


class SegmentationError(PipelineError):
    pass


class MappingError(PipelineError):
    pass


class EmissionError(PipelineError):
    pass


class ConstraintValidationError(PipelineError):
    pass


class HarmonizationError(PipelineError):
    pass


class VaultError(PipelineError):
    pass
