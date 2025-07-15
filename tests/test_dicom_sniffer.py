import pytest
import os
import shutil
from unittest.mock import patch, MagicMock
from datetime import datetime, date

from msk_io.retrieval.dicom_stream_sniffer import DICOMStreamSniffer
from msk_io.schema.retrieval_info import RetrievedDataInfo, DataSource
from msk_io.schema.dicom_data import DICOMVolume, DICOMPatientInfo, DICOMStudyInfo, DICOMSeriesInfo
from msk_io.errors import RetrievalError, DataValidationError


@pytest.fixture
def dicom_sniffer_instance(test_config):
    return DICOMStreamSniffer(test_config)


def test_dicom_sniffer_init(dicom_sniffer_instance, test_config):
    assert dicom_sniffer_instance.config == test_config
    assert dicom_sniffer_instance.base_url == test_config.retrieval.dicom_sniff_url
    assert dicom_sniffer_instance.download_dir == test_config.retrieval.data_download_dir
    assert os.path.exists(dicom_sniffer_instance.download_dir)


@pytest.mark.asyncio
async def test_discover_and_retrieve_studies_success(dicom_sniffer_instance, test_config):
    dummy_dicom_path = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "sample_data", "sample_dicom.dcm")
    assert os.path.exists(dummy_dicom_path)

    with patch('msk_io.retrieval.dicom_stream_sniffer.dcmread') as mock_dcmread, \
         patch('msk_io.retrieval.dicom_stream_sniffer.shutil.copy') as mock_copy:
        mock_ds = MagicMock()
        mock_ds.PatientID = "TEST_PATIENT"
        mock_ds.PatientName = "Test^Patient"
        mock_ds.StudyInstanceUID = "1.2.3.4.5"
        mock_ds.StudyID = "STUDY001"
        mock_ds.Modality = "CT"
        mock_ds.SeriesInstanceUID = "6.7.8.9.0"
        mock_ds.Rows = 512
        mock_ds.Columns = 512
        mock_ds.PixelSpacing = [0.5, 0.5]
        mock_ds.BitsAllocated = 16
        mock_ds.BitsStored = 12
        mock_ds.HighBit = 11
        mock_ds.PixelRepresentation = 0
        mock_ds.PhotometricInterpretation = "MONOCHROME2"
        mock_ds.TransferSyntaxUID = "1.2.840.10008.1.2.1"
        mock_ds.StudyDate = "20230101"
        mock_ds.PatientBirthDate = "19900510"
        mock_ds.StudyTime = "123456"
        mock_ds.PatientSex = "M"
        mock_dcmread.return_value = mock_ds
        mock_volume_path = os.path.join(dicom_sniffer_instance.download_dir, "dummy_volume.nii.gz")
        os.makedirs(os.path.dirname(mock_volume_path), exist_ok=True)
        with patch('builtins.open', MagicMock()):
            retrieval_info = await dicom_sniffer_instance.discover_and_retrieve_studies(patient_id="TEST_PATIENT")

        assert isinstance(retrieval_info, RetrievedDataInfo)
        assert retrieval_info.total_files_retrieved == 1
        assert retrieval_info.data_source.source_type == "Local_Filesystem"
        assert retrieval_info.retrieved_file_paths[0].endswith("sample_dicom.dcm")
        assert len(retrieval_info.series_volumes) == 1
        retrieved_volume = retrieval_info.series_volumes[0]
        assert isinstance(retrieved_volume, DICOMVolume)
        assert retrieved_volume.patient_info.patient_id == "TEST_PATIENT"
        assert retrieved_volume.study_info.study_instance_uid == "1.2.3.4.5"
        assert retrieved_volume.series_info.series_instance_uid == "6.7.8.9.0"
        assert mock_dcmread.called
        mock_copy.assert_called_once()


@pytest.mark.asyncio
async def test_discover_and_retrieve_studies_missing_dummy_dicom(dicom_sniffer_instance):
    dummy_dicom_path = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "sample_data", "sample_dicom.dcm")
    if os.path.exists(dummy_dicom_path):
        os.remove(dummy_dicom_path)

    with pytest.raises(RetrievalError):
        await dicom_sniffer_instance.discover_and_retrieve_studies()


@pytest.mark.asyncio
async def test_discover_and_retrieve_studies_invalid_dicom_format(dicom_sniffer_instance, test_config):
    invalid_dicom_path = os.path.join(test_config.retrieval.data_download_dir, "invalid.dcm")
    os.makedirs(os.path.dirname(invalid_dicom_path), exist_ok=True)
    with open(invalid_dicom_path, 'w') as f:
        f.write("NOT A DICOM FILE")

    original_dummy_dicom_path = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "sample_data", "sample_dicom.dcm")
    original_content = ""
    if os.path.exists(original_dummy_dicom_path):
        with open(original_dummy_dicom_path, 'r') as f:
            original_content = f.read()

    with open(original_dummy_dicom_path, 'w') as f:
        f.write("NOT A VALID DICOM FILE CONTENT")

    try:
        with pytest.raises(DataValidationError):
            await dicom_sniffer_instance.discover_and_retrieve_studies()
    finally:
        with open(original_dummy_dicom_path, 'w') as f:
            f.write(original_content)
        if os.path.exists(invalid_dicom_path):
            os.remove(invalid_dicom_path)

