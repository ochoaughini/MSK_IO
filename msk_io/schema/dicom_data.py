from typing import Dict, Any, List, Optional
from datetime import date, time
from pydantic import Field

from msk_io.schema._pydantic_base import MSKIOBaseModel


class DICOMPatientInfo(MSKIOBaseModel):
    """Information pertaining to the patient from DICOM metadata."""

    patient_id: str
    patient_name: str
    patient_sex: Optional[str] = None
    patient_birth_date: Optional[date] = None
    patient_age: Optional[str] = None  # e.g., '060Y'
    other_patient_ids: List[str] = Field(default_factory=list)


class DICOMStudyInfo(MSKIOBaseModel):
    """Information pertaining to a DICOM study."""

    study_instance_uid: str
    study_id: str
    study_description: Optional[str] = None
    study_date: Optional[date] = None
    study_time: Optional[time] = None
    accession_number: Optional[str] = None
    referring_physician_name: Optional[str] = None


class DICOMSeriesInfo(MSKIOBaseModel):
    """Information pertaining to a DICOM series."""

    series_instance_uid: str
    series_number: Optional[int] = None
    series_description: Optional[str] = None
    modality: str
    body_part_examined: Optional[str] = None
    protocol_name: Optional[str] = None


class DICOMImageInfo(MSKIOBaseModel):
    """Metadata specific to a single DICOM image slice or volume."""

    sop_instance_uid: str
    instance_number: Optional[int] = None
    pixel_spacing: Optional[List[float]] = None
    slice_thickness: Optional[float] = None
    image_orientation_patient: Optional[List[float]] = None
    image_position_patient: Optional[List[float]] = None
    rows: int
    columns: int
    bits_allocated: int
    bits_stored: int
    high_bit: int
    pixel_representation: int  # 0 for unsigned, 1 for signed
    window_center: Optional[float] = None
    window_width: Optional[float] = None
    rescale_intercept: Optional[float] = None
    rescale_slope: Optional[float] = None
    photometric_interpretation: str
    transfer_syntax_uid: str


class DICOMVolume(MSKIOBaseModel):
    """Represents a 3D medical image volume derived from DICOM series."""

    series_instance_uid: str
    dicom_files: List[str]
    volume_path: str
    original_modality: str
    patient_info: DICOMPatientInfo
    study_info: DICOMStudyInfo
    series_info: DICOMSeriesInfo
    volume_shape: List[int]
    voxel_spacing: List[float]


class DICOMData(MSKIOBaseModel):
    """Aggregated DICOM data, potentially including multiple studies or volumes."""

    raw_dicom_paths: List[str]
    patient_info: DICOMPatientInfo
    studies: List[DICOMStudyInfo]
    series_volumes: List[DICOMVolume]
    all_raw_metadata: List[Dict[str, Any]] = Field(default_factory=list)

