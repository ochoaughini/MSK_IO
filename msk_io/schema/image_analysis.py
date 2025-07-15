from typing import Dict, Any, List, Literal, Optional
from datetime import datetime
from uuid import uuid4
from pydantic import Field

from msk_io.schema._pydantic_base import MSKIOBaseModel
from msk_io.schema.dicom_data import DICOMVolume


class ImageMetaData(MSKIOBaseModel):
    """Basic metadata for a processed image."""

    original_path: str
    processed_path: Optional[str] = None
    image_format: str  # e.g., 'png', 'nifti', 'dcm'
    dimensions: List[int]  # e.g., [H, W] or [D, H, W]
    channels: Optional[int] = None
    voxel_spacing: Optional[List[float]] = None  # for 3D volumes


class RegionOfInterest(MSKIOBaseModel):
    """Defines a region of interest within an image or volume."""

    roi_id: str
    label: str  # e.g., 'Kidney', 'Tumor', 'Liver'
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
    """Result of an image segmentation operation."""

    source_volume: DICOMVolume
    segmentation_id: str
    segmented_at: datetime = Field(default_factory=datetime.now)
    regions_of_interest: List[RegionOfInterest]
    segmentation_method: str
    processed_image_meta: ImageMetaData


class ImageFeature(MSKIOBaseModel):
    """A generic feature extracted from an image or ROI."""

    feature_name: str
    value: Any
    unit: Optional[str] = None
    description: Optional[str] = None
    method_used: Optional[str] = None


class ImageAnalysisResult(MSKIOBaseModel):
    """Comprehensive result of an image analysis, including segmentation and features."""

    analysis_id: str = Field(default_factory=lambda: str(uuid4()))
    analyzed_volume: DICOMVolume
    segmentation_results: List[ImageSegmentationResult] = Field(default_factory=list)
    extracted_features: List[ImageFeature] = Field(default_factory=list)
    qualitative_observations: Optional[str] = None
    analysis_time: datetime = Field(default_factory=datetime.now)
    status: Literal["SUCCESS", "FAILURE", "PARTIAL_SUCCESS"] = "SUCCESS"
    errors: List[Dict[str, Any]] = Field(default_factory=list)

    def add_segmentation(self, seg_result: ImageSegmentationResult) -> None:
        self.segmentation_results.append(seg_result)

    def add_feature(self, feature: ImageFeature) -> None:
        self.extracted_features.append(feature)

