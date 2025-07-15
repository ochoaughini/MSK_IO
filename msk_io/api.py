import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from msk_io.io_pipeline import MSKIOPipeline
from msk_io.schema.base import PipelineStatus
from msk_io.schema.reports import DiagnosticReport
from msk_io.utils.log_config import get_logger
from msk_io.utils.decorators import handle_errors, log_method_entry_exit

logger = get_logger(__name__)


class MSKIOAPI:
    def __init__(self):
        self.pipeline = MSKIOPipeline()
        logger.info("MSK-IO API initialized.")

    @handle_errors
    @log_method_entry_exit
    async def process_medical_data(
        self,
        input_file_path: Optional[str] = None,
        remote_dicom_url: Optional[str] = None,
        patient_id: Optional[str] = None,
    ) -> PipelineStatus:
        logger.info(
            f"API call to process medical data: Input File: {input_file_path}, Remote URL: {remote_dicom_url}, Patient ID: {patient_id}"
        )
        return await self.pipeline.run_full_pipeline(
            input_file_path=input_file_path,
            remote_dicom_url=remote_dicom_url,
            patient_id=patient_id,
        )

    @handle_errors
    @log_method_entry_exit
    async def get_pipeline_status(self, pipeline_id: str) -> Optional[PipelineStatus]:
        logger.warning(
            f"API call to get pipeline status for ID: {pipeline_id}. This is a conceptual stub."
        )
        dummy_status = PipelineStatus(
            pipeline_id=pipeline_id,
            overall_status="COMPLETED_SUCCESS",
            overall_message=f"Simulated completion for pipeline {pipeline_id}",
            start_time=datetime.now() - timedelta(minutes=5),
            end_time=datetime.now(),
            total_duration_seconds=300.0,
            tasks_status=[],
        )
        logger.info(
            f"Simulated status for pipeline {pipeline_id}: {dummy_status.overall_status}"
        )
        return dummy_status

    @handle_errors
    @log_method_entry_exit
    async def get_diagnostic_report(
        self, report_path: str
    ) -> Optional[DiagnosticReport]:
        logger.info(f"API call to retrieve diagnostic report from: {report_path}")
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)
            return DiagnosticReport.model_validate(report_data)
        except FileNotFoundError:
            logger.error(f"Diagnostic report not found at: {report_path}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in report file {report_path}: {e}")
            raise MSKIOError(f"Failed to parse diagnostic report: {report_path}") from e
        except Exception as e:
            logger.error(
                f"Unexpected error retrieving report {report_path}: {e}", exc_info=True
            )
            raise MSKIOError(
                f"Failed to retrieve diagnostic report: {report_path}"
            ) from e
