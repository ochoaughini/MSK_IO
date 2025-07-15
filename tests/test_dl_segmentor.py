import pytest
import os
from unittest.mock import patch, MagicMock
from datetime import date

from msk_io.image_processing.dl_segmentor import DLSegmentor
from msk_io.schema.dicom_data import DICOMVolume, DICOMPatientInfo, DICOMStudyInfo, DICOMSeriesInfo
from msk_io.schema.image_analysis import ImageSegmentationResult
from msk_io.errors import ImageProcessingError, ConfigurationError, DataValidationError, ExternalServiceError


@pytest.fixture
def dl_segmentor_instance(test_config):
    return DLSegmentor(test_config)


def test_dl_segmentor_init(dl_segmentor_instance, test_config):
    assert dl_segmentor_instance.config == test_config
    assert dl_segmentor_instance.model_path == test_config.image_processing.default_segmentation_model_path
    assert dl_segmentor_instance.confidence_threshold == test_config.image_processing.segmentation_confidence_threshold
    assert os.path.exists(dl_segmentor_instance.cache_dir)


@pytest.mark.asyncio
async def test_dl_segmentor_load_model_success(dl_segmentor_instance):
    with patch.object(dl_segmentor_instance, '_model', new=None):
        model = dl_segmentor_instance._load_model()
        assert model == "DUMMY_LOADED_DL_MODEL"


@pytest.mark.asyncio
async def test_dl_segmentor_load_model_missing_config(dl_segmentor_instance):
    dl_segmentor_instance.config.image_processing.default_segmentation_model_path = None
    with pytest.raises(ConfigurationError):
        dl_segmentor_instance._load_model()


@pytest.mark.asyncio
async def test_dl_segmentor_segment_volume_success(dl_segmentor_instance, mock_dicom_volume):
    if not os.path.exists(mock_dicom_volume.volume_path):
        os.makedirs(os.path.dirname(mock_dicom_volume.volume_path), exist_ok=True)
        with open(mock_dicom_volume.volume_path, 'w') as f:
            f.write("DUMMY VOLUME DATA")
    with patch.object(dl_segmentor_instance, '_load_model', return_value="MOCKED_DL_MODEL"), \
         patch.object(dl_segmentor_instance, '_model', "MOCKED_DL_MODEL"):
        result = await dl_segmentor_instance.segment_image_volume(mock_dicom_volume)
        assert isinstance(result, ImageSegmentationResult)
        assert result.segmentation_method == "Deep Learning Inference"
        assert len(result.regions_of_interest) == 1
        assert result.regions_of_interest[0].confidence_score > dl_segmentor_instance.confidence_threshold
        assert os.path.exists(result.regions_of_interest[0].mask_file_path)


@pytest.mark.asyncio
async def test_dl_segmentor_segment_volume_missing_volume_file(dl_segmentor_instance, mock_dicom_volume):
    mock_dicom_volume.volume_path = "/path/to/nonexistent_volume.nii.gz"
    with pytest.raises(DataValidationError):
        await dl_segmentor_instance.segment_image_volume(mock_dicom_volume)


@pytest.mark.asyncio
async def test_dl_segmentor_segment_volume_model_loading_failure(dl_segmentor_instance, mock_dicom_volume):
    if not os.path.exists(mock_dicom_volume.volume_path):
        os.makedirs(os.path.dirname(mock_dicom_volume.volume_path), exist_ok=True)
        with open(mock_dicom_volume.volume_path, 'w') as f:
            f.write("DUMMY VOLUME DATA")
    with patch.object(dl_segmentor_instance, '_load_model', side_effect=ExternalServiceError("Model init failed")):
        with pytest.raises(ExternalServiceError):
            await dl_segmentor_instance.segment_image_volume(mock_dicom_volume)
    assert dl_segmentor_instance._model is None

