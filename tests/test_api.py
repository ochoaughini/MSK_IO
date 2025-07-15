import pytest
import os
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from msk_io.api import MSKIOAPI
from msk_io.schema.base import PipelineStatus
from msk_io.schema.reports import DiagnosticReport
from msk_io.errors import MSKIOError


@pytest.fixture
def api_instance(test_config):
    """Provides a fresh MSKIOAPI instance for each test."""
    with patch("msk_io.CONFIG", new=test_config):
        api = MSKIOAPI()
        api.pipeline.harmonizer.run_task_pipeline = AsyncMock()
        return api


@pytest.mark.asyncio
async def test_process_medical_data_local_success(
    api_instance, mock_dicom_volume, test_config
):
    """Tests successful pipeline processing with a local file."""
    dummy_input_path = mock_dicom_volume.volume_path
    patient_id = "TEST_P1_LOCAL"

    mock_pipeline_status = PipelineStatus(
        pipeline_id="mock_pipeline_id_local",
        overall_status="COMPLETED_SUCCESS",
        overall_message="Mock pipeline completed for local file.",
        final_report_path=os.path.join(
            test_config.app.output_directory, "mock_report_local.json"
        ),
    )
    api_instance.pipeline.harmonizer.run_task_pipeline.return_value = (
        mock_pipeline_status
    )

    dummy_report_data = DiagnosticReport(
        patient_info=mock_dicom_volume.patient_info,
        study_info=mock_dicom_volume.study_info,
        overall_conclusion="Simulated successful diagnosis for local.",
        diagnostic_findings=[],
    ).model_dump_json()
    os.makedirs(os.path.dirname(mock_pipeline_status.final_report_path), exist_ok=True)
    with open(mock_pipeline_status.final_report_path, "w") as f:
        f.write(dummy_report_data)

    status = await api_instance.process_medical_data(
        input_file_path=dummy_input_path, patient_id=patient_id
    )

    api_instance.pipeline.harmonizer.run_task_pipeline.assert_awaited_once_with(
        pipeline_task_definition=pytest.any_arg
    )
    assert status.overall_status == "COMPLETED_SUCCESS"
    assert status.pipeline_id == "mock_pipeline_id_local"
    assert status.final_report_path == mock_pipeline_status.final_report_path


@pytest.mark.asyncio
async def test_process_medical_data_remote_success(
    api_instance, mock_dicom_volume, test_config
):
    """Tests successful pipeline processing with a remote URL."""
    dummy_remote_url = "http://example.com/test.dcm"
    patient_id = "TEST_P2_REMOTE"

    mock_pipeline_status = PipelineStatus(
        pipeline_id="mock_pipeline_id_remote",
        overall_status="COMPLETED_SUCCESS",
        overall_message="Mock pipeline completed for remote URL.",
        final_report_path=os.path.join(
            test_config.app.output_directory, "mock_report_remote.json"
        ),
    )
    api_instance.pipeline.harmonizer.run_task_pipeline.return_value = (
        mock_pipeline_status
    )

    dummy_report_data = DiagnosticReport(
        patient_info=mock_dicom_volume.patient_info,
        study_info=mock_dicom_volume.study_info,
        overall_conclusion="Simulated successful diagnosis for remote.",
        diagnostic_findings=[],
    ).model_dump_json()
    os.makedirs(os.path.dirname(mock_pipeline_status.final_report_path), exist_ok=True)
    with open(mock_pipeline_status.final_report_path, "w") as f:
        f.write(dummy_report_data)

    status = await api_instance.process_medical_data(
        remote_dicom_url=dummy_remote_url, patient_id=patient_id
    )

    api_instance.pipeline.harmonizer.run_task_pipeline.assert_awaited_once()
    assert status.overall_status == "COMPLETED_SUCCESS"
    assert status.pipeline_id == "mock_pipeline_id_remote"
    assert status.final_report_path == mock_pipeline_status.final_report_path


@pytest.mark.asyncio
async def test_process_medical_data_failure(api_instance, mock_dicom_volume):
    """Tests pipeline processing failure."""
    dummy_input_path = mock_dicom_volume.volume_path
    api_instance.pipeline.harmonizer.run_task_pipeline.side_effect = MSKIOError(
        "Simulated pipeline failure."
    )

    with pytest.raises(MSKIOError, match="Simulated pipeline failure."):
        await api_instance.process_medical_data(input_file_path=dummy_input_path)

    api_instance.pipeline.harmonizer.run_task_pipeline.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_pipeline_status_found(api_instance):
    """Tests retrieving an existing pipeline status."""
    pipeline_id = "existing_pipeline_id"
    status = await api_instance.get_pipeline_status(pipeline_id)

    assert status is not None
    assert status.pipeline_id == pipeline_id
    assert (
        status.overall_status == "COMPLETED_SUCCESS"
    )  # Based on the API's dummy implementation


@pytest.mark.asyncio
async def test_get_diagnostic_report_success(
    api_instance, mock_dicom_volume, test_config
):
    """Tests successful retrieval of a diagnostic report."""
    report_path = os.path.join(
        test_config.app.output_directory, "test_report_success.json"
    )
    dummy_report = DiagnosticReport(
        patient_info=mock_dicom_volume.patient_info,
        study_info=mock_dicom_volume.study_info,
        overall_conclusion="This is a test report conclusion.",
        diagnostic_findings=[],
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(dummy_report.model_dump_json(indent=2))

    report = await api_instance.get_diagnostic_report(report_path)

    assert report is not None
    assert report.overall_conclusion == "This is a test report conclusion."
    assert report.patient_info.patient_id == mock_dicom_volume.patient_info.patient_id


@pytest.mark.asyncio
async def test_get_diagnostic_report_not_found(api_instance):
    """Tests retrieval of a non-existent diagnostic report."""
    report = await api_instance.get_diagnostic_report(
        "/path/to/nonexistent_report.json"
    )
    assert report is None


@pytest.mark.asyncio
async def test_get_diagnostic_report_invalid_json(api_instance, test_config):
    """Tests retrieval of a diagnostic report with invalid JSON content."""
    invalid_report_path = os.path.join(
        test_config.app.output_directory, "invalid_report.json"
    )
    os.makedirs(os.path.dirname(invalid_report_path), exist_ok=True)
    with open(invalid_report_path, "w") as f:
        f.write("This is not valid JSON {")

    with pytest.raises(MSKIOError, match="Failed to parse diagnostic report"):
        await api_instance.get_diagnostic_report(invalid_report_path)
