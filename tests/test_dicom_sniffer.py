import pytest
import os
import shutil
from unittest.mock import patch, MagicMock, AsyncMock
from msk_io.retrieval.dicom_stream_sniffer import DICOMStreamSniffer
from msk_io.schema.retrieval_info import RetrievedDataInfo, DataSource
from msk_io.schema.dicom_data import (
    DICOMVolume,
    DICOMPatientInfo,
    DICOMStudyInfo,
    DICOMSeriesInfo,
)
from msk_io.errors import RetrievalError, DataValidationError, ExternalServiceError
from datetime import datetime, date
import httpx


@pytest.fixture
def dicom_sniffer_instance(test_config):
    """Provides a DICOMStreamSniffer instance for testing."""
    return DICOMStreamSniffer(test_config)


def test_dicom_sniffer_init(dicom_sniffer_instance, test_config):
    """Test DICOMStreamSniffer initialization."""
    assert dicom_sniffer_instance.config == test_config
    assert dicom_sniffer_instance.base_url == test_config.retrieval.dicom_sniff_url
    assert (
        dicom_sniffer_instance.download_dir == test_config.retrieval.data_download_dir
    )
    assert os.path.exists(dicom_sniffer_instance.download_dir)


@pytest.mark.asyncio
async def test_discover_and_retrieve_studies_local_success(
    dicom_sniffer_instance, test_config
):
    """Tests successful (simulated) discovery and retrieval from a local path."""
    dummy_dicom_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "examples",
        "sample_data",
        "sample_dicom.dcm",
    )
    assert os.path.exists(dummy_dicom_path)

    with patch("msk_io.retrieval.dicom_stream_sniffer.dcmread") as mock_dcmread, patch(
        "msk_io.retrieval.dicom_stream_sniffer.shutil.copy"
    ) as mock_copy:
        mock_ds = MagicMock()
        mock_ds.PatientID = "TEST_PATIENT_LOCAL"
        mock_ds.PatientName = "Test^Patient"
        mock_ds.StudyInstanceUID = "1.2.3.4.5.LOCAL"
        mock_ds.StudyID = "STUDY001_LOCAL"
        mock_ds.Modality = "CT"
        mock_ds.SeriesInstanceUID = "6.7.8.9.0.LOCAL"
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

        mock_volume_path = os.path.join(
            dicom_sniffer_instance.download_dir, "dummy_volume.nii.gz"
        )
        os.makedirs(os.path.dirname(mock_volume_path), exist_ok=True)
        with patch("builtins.open", MagicMock()):
            retrieval_info = await dicom_sniffer_instance.discover_and_retrieve_studies(
                patient_id="TEST_PATIENT_LOCAL", local_dicom_path=dummy_dicom_path
            )

        assert isinstance(retrieval_info, RetrievedDataInfo)
        assert retrieval_info.total_files_retrieved == 1
        assert retrieval_info.data_source.source_type == "Local_Filesystem"
        assert retrieval_info.retrieved_file_paths[0].endswith("sample_dicom.dcm")
        assert (
            retrieval_info.series_volumes[0].patient_info.patient_id
            == "TEST_PATIENT_LOCAL"
        )
        assert mock_dcmread.called
        mock_copy.assert_called_once_with(
            dummy_dicom_path, retrieval_info.retrieved_file_paths[0]
        )


@pytest.mark.asyncio
async def test_discover_and_retrieve_studies_remote_success(
    dicom_sniffer_instance, test_config
):
    """Tests successful (simulated) discovery and retrieval from a remote URL."""
    remote_url = "http://example.com/remote_dicom.dcm"
    mock_response_content = b"DCM_FILE_CONTENT"

    with patch(
        "msk_io.retrieval.dicom_stream_sniffer.httpx.AsyncClient"
    ) as MockAsyncClient, patch(
        "msk_io.retrieval.dicom_stream_sniffer.dcmread"
    ) as mock_dcmread:
        mock_client_instance = AsyncMock()
        mock_response = AsyncMock()
        mock_response.content = mock_response_content
        mock_response.raise_for_status = MagicMock()
        mock_client_instance.get.return_value.__aenter__.return_value = mock_response
        MockAsyncClient.return_value.__aenter__.return_value = mock_client_instance

        mock_ds = MagicMock()
        mock_ds.PatientID = "TEST_PATIENT_REMOTE"
        mock_ds.PatientName = "Remote^Patient"
        mock_ds.StudyInstanceUID = "1.2.3.4.5.REMOTE"
        mock_ds.StudyID = "STUDY001_REMOTE"
        mock_ds.Modality = "MR"
        mock_ds.SeriesInstanceUID = "6.7.8.9.0.REMOTE"
        mock_ds.Rows = 256
        mock_ds.Columns = 256
        mock_ds.PixelSpacing = [1.0, 1.0]
        mock_ds.BitsAllocated = 16
        mock_ds.BitsStored = 12
        mock_ds.HighBit = 11
        mock_ds.PixelRepresentation = 0
        mock_ds.PhotometricInterpretation = "MONOCHROME2"
        mock_ds.TransferSyntaxUID = "1.2.840.10008.1.2.1"
        mock_ds.StudyDate = "20230201"
        mock_ds.PatientBirthDate = "19850101"
        mock_ds.StudyTime = "100000"
        mock_ds.PatientSex = "F"
        mock_dcmread.return_value = mock_ds

        with patch("builtins.open", MagicMock()):
            retrieval_info = await dicom_sniffer_instance.discover_and_retrieve_studies(
                patient_id="TEST_PATIENT_REMOTE", remote_dicom_url=remote_url
            )

        assert isinstance(retrieval_info, RetrievedDataInfo)
        assert retrieval_info.total_files_retrieved == 1
        assert retrieval_info.data_source.source_type == "Remote_URL"
        assert retrieval_info.retrieved_file_paths[0].endswith("remote_dicom.dcm")
        assert (
            retrieval_info.series_volumes[0].patient_info.patient_id
            == "TEST_PATIENT_REMOTE"
        )

        MockAsyncClient.assert_called_once()
        mock_client_instance.get.assert_awaited_once_with(remote_url)
        mock_response.raise_for_status.assert_called_once()
        mock_dcmread.called_once()


@pytest.mark.asyncio
async def test_discover_and_retrieve_studies_remote_download_failure(
    dicom_sniffer_instance,
):
    """Tests remote download failure (e.g., network error)."""
    remote_url = "http://bad.example.com/dicom.dcm"

    with patch(
        "msk_io.retrieval.dicom_stream_sniffer.httpx.AsyncClient"
    ) as MockAsyncClient:
        mock_client_instance = AsyncMock()
        mock_client_instance.get.side_effect = httpx.RequestError(
            "Simulated network error"
        )
        MockAsyncClient.return_value.__aenter__.return_value = mock_client_instance

        with pytest.raises(ExternalServiceError, match="Failed to download DICOM from"):
            await dicom_sniffer_instance.discover_and_retrieve_studies(
                remote_dicom_url=remote_url
            )


@pytest.mark.asyncio
async def test_discover_and_retrieve_studies_missing_local_dicom_and_no_remote(
    dicom_sniffer_instance,
):
    """Tests retrieval failure when local_dicom_path is missing and no remote URL is given, falling back to dummy."""
    dummy_dicom_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "examples",
        "sample_data",
        "sample_dicom.dcm",
    )
    if os.path.exists(dummy_dicom_path):
        os.remove(dummy_dicom_path)

    with pytest.raises(RetrievalError, match="Dummy DICOM file not found"):
        await dicom_sniffer_instance.discover_and_retrieve_studies()


@pytest.mark.asyncio
async def test_discover_and_retrieve_studies_invalid_dicom_format(
    dicom_sniffer_instance, test_config
):
    """Tests retrieval failure when the retrieved file (local or remote) is not a valid DICOM."""
    local_invalid_dicom_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "examples",
        "sample_data",
        "sample_dicom.dcm",
    )
    original_content = ""
    if os.path.exists(local_invalid_dicom_path):
        with open(local_invalid_dicom_path, "rb") as f:
            original_content = f.read()

    with open(local_invalid_dicom_path, "w") as f:
        f.write("NOT A VALID DICOM FILE CONTENT")

    try:
        with pytest.raises(
            DataValidationError, match="Retrieved file is not a valid DICOM"
        ):
            await dicom_sniffer_instance.discover_and_retrieve_studies(
                local_dicom_path=local_invalid_dicom_path
            )
    finally:
        with open(local_invalid_dicom_path, "wb") as f:
            f.write(original_content)
