import os
from typing import Optional
from uuid import uuid4

from msk_io.schema.base import PipelineStatus, FileInfo
from msk_io.schema.task_definitions import TaskDefinition, AgentInstruction
from msk_io.control.multi_agent_harmonizer import MultiAgentHarmonizer
from msk_io.errors import MSKIOError
from msk_io.utils.log_config import get_logger
from msk_io.utils.decorators import handle_errors, log_method_entry_exit
from msk_io import CONFIG

logger = get_logger(__name__)


class MSKIOPipeline:
    """The main pipeline orchestrator for the MSK-IO diagnostic process."""

    def __init__(self):
        if CONFIG is None:
            raise MSKIOError("Application configuration not loaded. Cannot initialize pipeline.")
        self.harmonizer = MultiAgentHarmonizer(CONFIG)
        self.config = CONFIG
        logger.info("MSK-IO Pipeline initialized.")

    @handle_errors
    @log_method_entry_exit
    async def run_full_pipeline(self, input_file_path: str, patient_id: Optional[str] = None) -> PipelineStatus:
        logger.info(f"Starting full pipeline for input file: {input_file_path}")
        file_info = FileInfo(
            file_path=input_file_path,
            file_name=os.path.basename(input_file_path),
            file_extension=os.path.splitext(input_file_path)[1],
        )

        task_id = str(uuid4())
        pipeline_task_definition = TaskDefinition(
            task_id=task_id,
            task_name=f"Full_Diagnostic_Pipeline_{os.path.basename(input_file_path)}",
            description=f"Process {input_file_path} for diagnostic assessment.",
            required_inputs=["DICOM_FILE"],
            output_type="DiagnosticReport",
            sequence_of_instructions=[],
        )

        effective_patient_id = patient_id if patient_id else "ANONYMOUS_PATIENT"

        pipeline_task_definition.sequence_of_instructions.append(
            AgentInstruction(
                command="RETRIEVE_DICOM_STUDY",
                parameters={"patient_id": effective_patient_id, "study_uid": f"simulated_uid_{task_id}", "input_file_path": input_file_path},
                target_agent="retrieval",
            )
        )

        pipeline_task_definition.sequence_of_instructions.append(
            AgentInstruction(
                command="PERFORM_DL_SEGMENTATION",
                parameters={"dicom_volume": {"$from_context": "retrieval_info.series_volumes.0"}},
                target_agent="image_processing",
            )
        )

        pipeline_task_definition.sequence_of_instructions.append(
            AgentInstruction(
                command="ANALYZE_WITH_LLM",
                parameters={
                    "agent_type": self.config.llm.default_llm_model,
                    "prompt_template_name": "DiagnosticAssessment",
                    "context_data": {
                        "patient_info_summary": {"$from_context": "retrieval_info.patient_info"},
                        "image_analysis_summary": {"$from_context": "segmentation_result.regions_of_interest"},
                        "clinical_context": "No specific clinical notes available.",
                    },
                },
                target_agent="llm_inference",
            )
        )

        pipeline_task_definition.sequence_of_instructions.append(
            AgentInstruction(
                command="INDEX_DOCUMENT",
                parameters={
                    "doc_id": {"$from_context": "llm_analysis_result.analysis_id"},
                    "text_content": {"$from_context": "llm_analysis_result.summary"},
                    "metadata": {"source_file": input_file_path},
                },
                target_agent="semantic_indexing",
            )
        )

        pipeline_task_definition.sequence_of_instructions.append(
            AgentInstruction(
                command="GENERATE_DIAGNOSTIC_REPORT",
                parameters={
                    "patient_info": {"$from_context": "retrieval_info.patient_info"},
                    "study_info": {"$from_context": "retrieval_info.studies.0"},
                    "diagnostic_findings": {"$from_context": "llm_analysis_result.extracted_findings"},
                    "image_summaries_json": {"$from_context": "segmentation_result"},
                    "llm_summaries_json": {"$from_context": "llm_analysis_result"},
                },
                target_agent="reporting",
            )
        )

        pipeline_status = await self.harmonizer.run_task_pipeline(pipeline_task_definition)

        if pipeline_status.overall_status == "COMPLETED_SUCCESS":
            logger.info(f"Full pipeline for {input_file_path} completed successfully.")
        else:
            logger.error(f"Full pipeline for {input_file_path} finished with status: {pipeline_status.overall_status}")

        return pipeline_status

