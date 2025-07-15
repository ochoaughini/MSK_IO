import pytest
from click.testing import CliRunner
import os
import json
from unittest.mock import patch, MagicMock, AsyncMock
import logging
import asyncio

from msk_io.cli import cli
from msk_io.schema.base import PipelineStatus
from msk_io.schema.reports import DiagnosticReport
from msk_io.errors import MSKIOError


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def mock_api_and_config(test_config):
    mock_pipeline_status = PipelineStatus(
        pipeline_id="mock_pipeline_id",
        overall_status="COMPLETED_SUCCESS",
        overall_message="Mock pipeline completed successfully.",
        final_report_path=os.path.join(test_config.app.output_directory, "mock_cli_report.json"),
    )

    mock_diagnostic_report = DiagnosticReport(
        patient_info=MagicMock(patient_id="CLI_P1"),
        study_info=MagicMock(study_instance_uid="CLI_S1"),
        overall_conclusion="Mock report conclusion from CLI test.",
        diagnostic_findings=[],
    )

    os.makedirs(os.path.dirname(mock_pipeline_status.final_report_path), exist_ok=True)
    with open(mock_pipeline_status.final_report_path, 'w') as f:
        f.write(mock_diagnostic_report.model_dump_json(indent=2))

    mock_api = MagicMock()
    mock_api.process_medical_data = AsyncMock(return_value=mock_pipeline_status)
    mock_api.get_pipeline_status = AsyncMock(return_value=mock_pipeline_status)
    mock_api.get_diagnostic_report = AsyncMock(return_value=mock_diagnostic_report)

    with patch('msk_io.CONFIG', new=test_config), patch('msk_io.api.MSKIOAPI', return_value=mock_api) as MockMSKIOAPI:
        yield runner, mock_api, test_config


def test_cli_process_success(runner, mock_api_and_config):
    runner, mock_api, test_config = mock_api_and_config
    dummy_input_file = os.path.join(test_config.app.watch_directory, "dummy_input.dcm")
    os.makedirs(os.path.dirname(dummy_input_file), exist_ok=True)
    with open(dummy_input_file, 'w') as f:
        f.write("dummy content")

    result = runner.invoke(cli, ['process', dummy_input_file, '--patient-id', 'P123'])

    assert result.exit_code == 0
    assert "Pipeline ID: mock_pipeline_id" in result.output
    assert "Status: COMPLETED_SUCCESS" in result.output
    assert "Final Report Path:" in result.output
    assert "Overall Conclusion: Mock report conclusion from CLI test." in result.output
    mock_api.process_medical_data.assert_called_once_with(dummy_input_file, 'P123')
    mock_api.get_diagnostic_report.assert_awaited_once()


def test_cli_process_failure(runner, mock_api_and_config):
    runner, mock_api, test_config = mock_api_and_config
    dummy_input_file = os.path.join(test_config.app.watch_directory, "dummy_input.dcm")
    os.makedirs(os.path.dirname(dummy_input_file), exist_ok=True)
    with open(dummy_input_file, 'w') as f:
        f.write("dummy content")

    mock_api.process_medical_data.side_effect = MSKIOError("Simulated pipeline failure")

    result = runner.invoke(cli, ['process', dummy_input_file])

    assert result.exit_code == 1
    assert "Error: Pipeline processing failed: Simulated pipeline failure" in result.output
    mock_api.process_medical_data.assert_called_once_with(dummy_input_file, None)


def test_cli_monitor_success(runner, mock_api_and_config):
    runner, mock_api, test_config = mock_api_and_config
    mock_monitor_instance = MagicMock()
    mock_monitor_instance.start_monitoring = MagicMock()
    mock_monitor_instance.stop_monitoring = MagicMock()

    with patch('msk_io.cli.DirectoryMonitor', return_value=mock_monitor_instance), \
         patch('time.sleep', side_effect=KeyboardInterrupt), \
         patch('asyncio.new_event_loop'), \
         patch('asyncio.set_event_loop'), \
         patch('asyncio.run'):
        result = runner.invoke(cli, ['monitor', '-i', '1'])
        assert "Monitoring directory:" in result.output
        assert "Press Ctrl+C to stop monitoring." in result.output
        assert "Stopping monitoring." in result.output
        assert "Monitoring stopped." in result.output
        mock_monitor_instance.start_monitoring.assert_called_once()
        mock_monitor_instance.stop_monitoring.assert_called_once()
        assert result.exit_code == 0


def test_cli_status_found(runner, mock_api_and_config):
    runner, mock_api, test_config = mock_api_and_config
    result = runner.invoke(cli, ['status', '--pipeline-id', 'mock_pipeline_id'])

    assert result.exit_code == 0
    assert "--- Pipeline Status for ID: mock_pipeline_id ---" in result.output
    assert "Overall Status: COMPLETED_SUCCESS" in result.output
    mock_api.get_pipeline_status.assert_called_once_with('mock_pipeline_id')


def test_cli_status_not_found(runner, mock_api_and_config):
    runner, mock_api, test_config = mock_api_and_config
    mock_api.get_pipeline_status.return_value = None
    result = runner.invoke(cli, ['status', '--pipeline-id', 'non_existent_id'])

    assert result.exit_code == 0
    assert "Pipeline with ID 'non_existent_id' not found or status not available." in result.output
    mock_api.get_pipeline_status.assert_called_once_with('non_existent_id')


def test_cli_config_display(runner, mock_api_and_config):
    runner, mock_api, test_config = mock_api_and_config
    result = runner.invoke(cli, ['config'])

    assert result.exit_code == 0
    config_output = json.loads(result.output)
    assert "app" in config_output
    assert "llm" in config_output
    assert "openai_api_key" in config_output["llm"]
    assert config_output["llm"]["openai_api_key"] == "********"
    assert os.path.isdir(config_output["app"]["watch_directory"])
    assert "watch" in config_output["app"]["watch_directory"]
    assert "pytest_temp" in config_output["app"]["watch_directory"]


def test_cli_log_level_override(runner, mock_api_and_config):
    runner, mock_api, test_config = mock_api_and_config
    with patch('msk_io.cli.setup_logging') as mock_setup_logging:
        result = runner.invoke(cli, ['--log-level', 'DEBUG', 'config'])
        assert result.exit_code == 0
        mock_setup_logging.assert_called_once()
        args, kwargs = mock_setup_logging.call_args
        assert kwargs['level'] == logging.DEBUG

