import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from msk_io.schema.base import PipelineStatus, TaskStatus
from msk_io.schema.task_definitions import AgentInstruction, AgentResponse, TaskDefinition
from msk_io.schema.reports import DiagnosticReport
from msk_io.schema.synthesis import MultiModalSynthesisResult
from msk_io.errors import AgentOrchestrationError, ProcessingError
from msk_io.utils.log_config import get_logger
from msk_io.utils.decorators import handle_errors, log_method_entry_exit

logger = get_logger(__name__)

class Agent:
    def __init__(self, name: str, config):
        self.name = name
        self.config = config
        self.logger = get_logger(f"agent.{self.name}")

    async def execute_instruction(self, instruction: AgentInstruction, pipeline_context: Dict[str, Any]) -> AgentResponse:
        self.logger.info(f"Agent '{self.name}' received instruction: {instruction.command}")
        try:
            result_data = await self._process_command(instruction.command, instruction.parameters, pipeline_context)
            return AgentResponse(instruction_id=instruction.instruction_id, agent_name=self.name, status="SUCCESS", output_data=result_data)
        except Exception as e:
            self.logger.error(f"Agent '{self.name}' failed to execute instruction {instruction.command}: {e}", exc_info=True)
            return AgentResponse(instruction_id=instruction.instruction_id, agent_name=self.name, status="FAILED", error_message=str(e), output_data={"error_type": e.__class__.__name__})

    async def _process_command(self, command: str, parameters: Dict[str, Any], pipeline_context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement _process_command.")

class RetrievalAgent(Agent):
    def __init__(self, config):
        super().__init__("RetrievalAgent", config)
        from msk_io.retrieval.dicom_stream_sniffer import DICOMStreamSniffer
        from msk_io.retrieval.ohif_canvas_extractor import OHIFCanvasExtractor
        self.dicom_sniffer = DICOMStreamSniffer(config)
        self.ohif_extractor = OHIFCanvasExtractor(config)

    async def _process_command(self, command: str, parameters: Dict[str, Any], pipeline_context: Dict[str, Any]) -> Dict[str, Any]:
        if command == "RETRIEVE_DICOM_STUDY":
            patient_id = parameters.get("patient_id")
            study_uid = parameters.get("study_uid")
            retrieval_info = self.dicom_sniffer.discover_and_retrieve_studies(patient_id, study_uid)
            return {"retrieval_info": retrieval_info.model_dump()}
        elif command == "EXTRACT_OHIF_IMAGES":
            ohif_url = parameters["ohif_url"]
            study_id = parameters["study_id"]
            series_id = parameters.get("series_id")
            retrieval_info = await self.ohif_extractor.extract_images_from_ohif(ohif_url, study_id, series_id)
            return {"retrieval_info": retrieval_info.model_dump()}
        else:
            raise ValueError(f"Unknown command for RetrievalAgent: {command}")

class ImageProcessingAgent(Agent):
    def __init__(self, config):
        super().__init__("ImageProcessingAgent", config)
        from msk_io.image_processing.segmentor import Segmentor
        from msk_io.image_processing.dl_segmentor import DLSegmentor
        from msk_io.image_processing.totalsegmentatorv2 import TotalSegmentatorV2
        self.basic_segmentor = Segmentor(config)
        self.dl_segmentor = DLSegmentor(config)
        self.total_segmentator = TotalSegmentatorV2(config)

    async def _process_command(self, command: str, parameters: Dict[str, Any], pipeline_context: Dict[str, Any]) -> Dict[str, Any]:
        if command == "PERFORM_BASIC_SEGMENTATION":
            dicom_volume_data = parameters["dicom_volume"]
            dicom_volume = DICOMVolume.model_validate(dicom_volume_data)
            threshold = parameters.get("threshold")
            segmentation_result = self.basic_segmentor.segment_image_volume(dicom_volume, threshold)
            return {"segmentation_result": segmentation_result.model_dump()}
        elif command == "PERFORM_DL_SEGMENTATION":
            dicom_volume_data = parameters["dicom_volume"]
            dicom_volume = DICOMVolume.model_validate(dicom_volume_data)
            segmentation_result = self.dl_segmentor.segment_image_volume(dicom_volume)
            return {"segmentation_result": segmentation_result.model_dump()}
        elif command == "RUN_TOTALSEGMENTATOR":
            dicom_volume_data = parameters["dicom_volume"]
            dicom_volume = DICOMVolume.model_validate(dicom_volume_data)
            tasks = parameters.get("tasks")
            segmentation_result = self.total_segmentator.run_segmentation(dicom_volume, tasks)
            return {"segmentation_result": segmentation_result.model_dump()}
        else:
            raise ValueError(f"Unknown command for ImageProcessingAgent: {command}")

    from msk_io.schema.dicom_data import DICOMVolume

class LLMInferenceAgent(Agent):
    def __init__(self, config):
        super().__init__("LLMInferenceAgent", config)
        from msk_io.inference.llm_agents import LLMAgentFactory
        from msk_io.inference.prompts.agent_prompts import get_prompt_template
        self.llm_agent_factory = LLMAgentFactory
        self.get_prompt_template = get_prompt_template

    async def _process_command(self, command: str, parameters: Dict[str, Any], pipeline_context: Dict[str, Any]) -> Dict[str, Any]:
        if command == "ANALYZE_WITH_LLM":
            agent_type = parameters.get("agent_type", self.config.llm.default_llm_model)
            prompt_template_name = parameters["prompt_template_name"]
            context_data = parameters["context_data"]
            llm_agent = self.llm_agent_factory.get_agent(agent_type, self.config)
            prompt_template = self.get_prompt_template(prompt_template_name)
            llm_analysis_result = llm_agent.analyze_data(prompt_template, context_data)
            return {"llm_analysis_result": llm_analysis_result.model_dump()}
        else:
            raise ValueError(f"Unknown command for LLMInferenceAgent: {command}")

class SemanticIndexingAgent(Agent):
    def __init__(self, config):
        super().__init__("SemanticIndexingAgent", config)
        from msk_io.indexer.semantic_indexer import SemanticIndexer
        self.semantic_indexer = SemanticIndexer(config)

    async def _process_command(self, command: str, parameters: Dict[str, Any], pipeline_context: Dict[str, Any]) -> Dict[str, Any]:
        if command == "INDEX_DOCUMENT":
            doc_id = parameters["doc_id"]
            text_content = parameters["text_content"]
            metadata = parameters.get("metadata")
            self.semantic_indexer.index_document(doc_id, text_content, metadata)
            return {"status": "indexed", "doc_id": doc_id}
        elif command == "QUERY_INDEX":
            query_text = parameters["query_text"]
            top_k = parameters.get("top_k", 5)
            results = self.semantic_indexer.query_semantic_index(query_text, top_k)
            return {"query_results": results}
        else:
            raise ValueError(f"Unknown command for SemanticIndexingAgent: {command}")

class ReportingAgent(Agent):
    def __init__(self, config):
        super().__init__("ReportingAgent", config)
        from msk_io.schema.reports import DiagnosticReport
        from msk_io.inference.prompts.agent_prompts import get_prompt_template
        from msk_io.inference.llm_agents import LLMAgentFactory
        self.get_prompt_template = get_prompt_template
        self.llm_agent_factory = LLMAgentFactory

    async def _process_command(self, command: str, parameters: Dict[str, Any], pipeline_context: Dict[str, Any]) -> Dict[str, Any]:
        if command == "GENERATE_DIAGNOSTIC_REPORT":
            patient_info_data = parameters["patient_info"]
            study_info_data = parameters["study_info"]
            findings_list_data = parameters["diagnostic_findings"]
            image_summaries_data = parameters["image_analysis_summaries"]
            llm_summaries_data = parameters["llm_analysis_summaries"]
            from msk_io.schema.dicom_data import DICOMPatientInfo, DICOMStudyInfo
            from msk_io.schema.llm_output import DiagnosticFinding, LLMAnalysisResult
            from msk_io.schema.image_analysis import ImageAnalysisResult
            patient_info = DICOMPatientInfo.model_validate(patient_info_data)
            study_info = DICOMStudyInfo.model_validate(study_info_data)
            diagnostic_findings = [DiagnosticFinding.model_validate(f) for f in findings_list_data]
            image_analysis_summaries = [ImageAnalysisResult.model_validate(i) for i in image_summaries_data]
            llm_analysis_summaries = [LLMAnalysisResult.model_validate(l) for l in llm_summaries_data]
            report_template = self.get_prompt_template("ReportGeneration")
            llm_agent = self.llm_agent_factory.get_agent(self.config.llm.default_llm_model, self.config)
            report_context = {
                "patient_details_json": patient_info.model_dump_json(),
                "study_details_json": study_info.model_dump_json(),
                "findings_list_json": [f.model_dump_json() for f in diagnostic_findings],
                "image_summaries_json": [i.model_dump_json() for i in image_analysis_summaries],
                "llm_summaries_json": [l.model_dump_json() for l in llm_analysis_summaries]
            }
            llm_report_result = llm_agent.analyze_data(report_template, report_context)
            try:
                generated_report_dict = json.loads(llm_report_result.llm_output.choices[0].text)
                final_report = DiagnosticReport(
                    patient_info=patient_info,
                    study_info=study_info,
                    overall_conclusion=generated_report_dict.get("overall_conclusion", llm_report_result.summary),
                    diagnostic_findings=diagnostic_findings,
                    associated_volumes=generated_report_dict.get("associated_volumes", []),
                    image_analysis_summaries=image_analysis_summaries,
                    llm_analysis_summaries=llm_analysis_summaries,
                    recommendations=generated_report_dict.get("recommendations", []),
                    status="PRELIMINARY",
                    reviewer_notes=f"Generated by LLM Agent: {self.config.llm.default_llm_model}"
                )
            except json.JSONDecodeError as e:
                logger.error(f"LLM did not return valid JSON for report: {e}. Falling back to basic report.")
                final_report = DiagnosticReport(
                    patient_info=patient_info,
                    study_info=study_info,
                    overall_conclusion=llm_report_result.summary,
                    diagnostic_findings=diagnostic_findings,
                    image_analysis_summaries=image_analysis_summaries,
                    llm_analysis_summaries=llm_analysis_summaries,
                    recommendations=["Review LLM output for structured report generation."],
                    status="ERROR"
                )
            report_filename = f"report_{patient_info.patient_id}_{study_info.study_instance_uid}.json"
            report_path = os.path.join(self.config.app.output_directory, report_filename)
            try:
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(final_report.model_dump_json(indent=2))
                logger.info(f"Diagnostic report saved to: {report_path}")
                final_report.status = "FINAL"
                final_report.attachments.append({"file_path": report_path, "description": "Generated JSON Report", "mime_type": "application/json"})
            except Exception as e:
                logger.error(f"Failed to save diagnostic report: {e}")
                raise ProcessingError(f"Failed to save diagnostic report: {e}") from e
            return {"report_path": report_path, "report_summary": final_report.model_dump()}
        else:
            raise ValueError(f"Unknown command for ReportingAgent: {command}")

class MultiAgentHarmonizer:
    def __init__(self, config):
        self.config = config
        self.agents: Dict[str, Agent] = {
            "retrieval": RetrievalAgent(config),
            "image_processing": ImageProcessingAgent(config),
            "llm_inference": LLMInferenceAgent(config),
            "semantic_indexing": SemanticIndexingAgent(config),
            "reporting": ReportingAgent(config)
        }
        self.logger = get_logger("MultiAgentHarmonizer")
        self.logger.info("Multi-Agent Harmonizer initialized with agents: " + ", ".join(self.agents.keys()))

    @handle_errors
    @log_method_entry_exit
    async def run_task_pipeline(self, task_definition: TaskDefinition) -> PipelineStatus:
        pipeline_status = PipelineStatus(overall_status="RUNNING", pipeline_id=task_definition.task_id, overall_message=f"Starting pipeline for task: {task_definition.task_name}")
        pipeline_context: Dict[str, Any] = {}
        self.logger.info(f"Starting pipeline '{task_definition.task_name}' (ID: {task_definition.task_id})...")
        for instruction in task_definition.sequence_of_instructions:
            task_status = TaskStatus(task_id=instruction.instruction_id, task_name=f"{instruction.command} by {instruction.target_agent}", status="IN_PROGRESS")
            pipeline_status.add_task_status(task_status)
            self.logger.info(f"Executing instruction: {instruction.command} (Agent: {instruction.target_agent})...")
            try:
                if instruction.target_agent not in self.agents:
                    raise AgentOrchestrationError(f"Target agent '{instruction.target_agent}' not found.")
                agent = self.agents[instruction.target_agent]
                response = await agent.execute_instruction(instruction, pipeline_context)
                if response.status == "SUCCESS":
                    task_status.update_status("COMPLETED", message="Instruction completed successfully.")
                    pipeline_context.update(response.output_data)
                    self.logger.info(f"Instruction {instruction.command} completed successfully.")
                else:
                    task_status.update_status("FAILED", message=response.error_message, error_details={"agent_response": response.model_dump()})
                    self.logger.error(f"Instruction {instruction.command} failed: {response.error_message}")
                    pipeline_status.finalize_pipeline("COMPLETED_WITH_ERRORS", message=f"Pipeline completed with errors. Task '{instruction.command}' failed.", fatal_error=task_status.error_details)
                    return pipeline_status
            except Exception as e:
                task_status.update_status("FAILED", message=f"Unexpected error: {e}", error_details={"exception": str(e), "type": e.__class__.__name__})
                self.logger.critical(f"Critical failure in pipeline task '{instruction.command}': {e}", exc_info=True)
                pipeline_status.finalize_pipeline("FAILED", message=f"Pipeline failed critically during task '{instruction.command}': {e}", fatal_error=task_status.error_details)
                return pipeline_status
            pipeline_status.add_task_status(task_status)
        pipeline_status.finalize_pipeline("COMPLETED_SUCCESS", message="All pipeline tasks completed successfully.")
        self.logger.info(f"Pipeline '{task_definition.task_name}' completed successfully.")
        return pipeline_status
