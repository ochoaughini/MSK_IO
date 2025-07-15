from typing import Dict, Any, List, Optional
from datetime import date, time
from pydantic import Field
from msk_io.schema._pydantic_base import MSKIOBaseModel

class DICOMPatientInfo(MSKIOBaseModel):
    patient_id: str
    patient_name: str
    patient_sex: Optional[str] = None
    patient_birth_date: Optional[date] = None
    patient_age: Optional[str] = None
    other_patient_ids: List[str] = []

class DICOMStudyInfo(MSKIOBaseModel):
    study_instance_uid: str
    study_id: str
    study_description: Optional[str] = None
    study_date: Optional[date] = None
    study_time: Optional[time] = None
    accession_number: Optional[str] = None
    referring_physician_name: Optional[str] = None

class DICOMSeriesInfo(MSKIOBaseModel):
    series_instance_uid: str
    series_number: Optional[int] = None
    series_description: Optional[str] = None
    modality: str
    body_part_examined: Optional[str] = None
    protocol_name: Optional[str] = None

class DICOMImageInfo(MSKIOBaseModel):
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
    pixel_representation: int
    window_center: Optional[float] = None
    window_width: Optional[float] = None
    rescale_intercept: Optional[float] = None
    rescale_slope: Optional[float] = None
    photometric_interpretation: str
    transfer_syntax_uid: str

class DICOMVolume(MSKIOBaseModel):
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
    raw_dicom_paths: List[str]
    patient_info: DICOMPatientInfo
    studies: List[DICOMStudyInfo]
    series_volumes: List[DICOMVolume]
    all_raw_metadata: List[Dict[str, Any]] = []
