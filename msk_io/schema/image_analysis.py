from datetime import datetime
from typing import Dict, Any, List, Literal, Optional
from uuid import uuid4
from pydantic import Field
from msk_io.schema._pydantic_base import MSKIOBaseModel
from msk_io.schema.dicom_data import DICOMVolume

class ImageMetaData(MSKIOBaseModel):
    original_path: str
    processed_path: Optional[str] = None
    image_format: str
    dimensions: List[int]
    channels: Optional[int] = None
    voxel_spacing: Optional[List[float]] = None

class RegionOfInterest(MSKIOBaseModel):
    roi_id: str
    label: str
    bounding_box_2d: Optional[List[int]] = None
    bounding_box_3d: Optional[List[int]] = None
    centroid_2d: Optional[List[float]] = None
    centroid_3d: Optional[List[float]] = None
    volume_mm3: Optional[float] = None
    area_mm2: Optional[float] = None
    pixel_count: Optional[int] = None
    mask_file_path: Optional[str] = None
    confidence_score: Optional[float] = None
    segmentation_model_used: Optional[str] = None

class ImageSegmentationResult(MSKIOBaseModel):
    source_volume: DICOMVolume
    segmentation_id: str
    segmented_at: datetime = datetime.now()
    regions_of_interest: List[RegionOfInterest]
    segmentation_method: str
    processed_image_meta: ImageMetaData

class ImageFeature(MSKIOBaseModel):
    feature_name: str
    value: Any
    unit: Optional[str] = None
    description: Optional[str] = None
    method_used: Optional[str] = None

class ImageAnalysisResult(MSKIOBaseModel):
    analysis_id: str = Field(default_factory=lambda: str(uuid4()))
    analyzed_volume: DICOMVolume
    segmentation_results: List[ImageSegmentationResult] = []
    extracted_features: List[ImageFeature] = []
    qualitative_observations: Optional[str] = None
    analysis_time: datetime = datetime.now()
    status: Literal["SUCCESS", "FAILURE", "PARTIAL_SUCCESS"] = "SUCCESS"
    errors: List[Dict[str, Any]] = []

    def add_segmentation(self, seg_result: ImageSegmentationResult) -> None:
        self.segmentation_results.append(seg_result)

    def add_feature(self, feature: ImageFeature) -> None:
        self.extracted_features.append(feature)
