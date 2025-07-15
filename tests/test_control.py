import pytest
import asyncio
import os
from unittest.mock import MagicMock, AsyncMock, patch
import json
from datetime import datetime, date

from msk_io.control.multi_agent_harmonizer import MultiAgentHarmonizer, AgentResponse, RetrievalAgent, ImageProcessingAgent, LLMInferenceAgent, SemanticIndexingAgent, ReportingAgent
from msk_io.schema.task_definitions import TaskDefinition, AgentInstruction
from msk_io.schema.base import PipelineStatus
from msk_io.schema.dicom_data import DICOMVolume, DICOMPatientInfo, DICOMStudyInfo, DICOMSeriesInfo
from msk_io.schema.image_analysis import ImageSegmentationResult, RegionOfInterest, ImageMetaData
from msk_io.schema.llm_output import LLMAnalysisResult, LLMOutput, LLMResponseChoice, DiagnosticFinding
from msk_io.schema.reports import DiagnosticReport
from msk_io.schema.retrieval_info import RetrievedDataInfo, DataSource


@pytest.fixture
def harmonizer_instance(test_config):
    with patch('msk_io.control.multi_agent_harmonizer.RetrievalAgent', autospec=True) as MockRetrievalAgent, \
         patch('msk_io.control.multi_agent_harmonizer.ImageProcessingAgent', autospec=True) as MockImageProcessingAgent, \
         patch('msk_io.control.multi_agent_harmonizer.LLMInferenceAgent', autospec=True) as MockLLMInferenceAgent, \
         patch('msk_io.control.multi_agent_harmonizer.SemanticIndexingAgent', autospec=True) as MockSemanticIndexingAgent, \
         patch('msk_io.control.multi_agent_harmonizer.ReportingAgent', autospec=True) as MockReportingAgent:

        mock_retrieval_agent = MockRetrievalAgent.return_value
        mock_image_processing_agent = MockImageProcessingAgent.return_value
        mock_llm_inference_agent = MockLLMInferenceAgent.return_value
        mock_semantic_indexing_agent = MockSemanticIndexingAgent.return_value
        mock_reporting_agent = MockReportingAgent.return_value

        harmonizer = MultiAgentHarmonizer(test_config)
        harmonizer.agents = {
            "retrieval": mock_retrieval_agent,
            "image_processing": mock_image_processing_agent,
            "llm_inference": mock_llm_inference_agent,
            "semantic_indexing": mock_semantic_indexing_agent,
            "reporting": mock_reporting_agent,
        }
        yield harmonizer, {
            "retrieval": mock_retrieval_agent,
            "image_processing": mock_image_processing_agent,
            "llm_inference": mock_llm_inference_agent,
            "semantic_indexing": mock_semantic_indexing_agent,
            "reporting": mock_reporting_agent,
        }


@pytest.mark.asyncio
async def test_run_task_pipeline_success(harmonizer_instance, mock_dicom_volume, test_config):
    harmonizer, mocks = harmonizer_instance
    patient_info = mock_dicom_volume.patient_info
    study_info = mock_dicom_volume.study_info

    retrieval_info_data = RetrievedDataInfo(
        retrieval_id="ret_123",
        data_source=DataSource(source_id="test", source_type="Local_Filesystem"),
        original_query="test_patient",
        retrieved_file_paths=[mock_dicom_volume.volume_path],
        total_files_retrieved=1,
        total_size_bytes=1000,
        retrieval_start_time=datetime.now(),
        retrieval_end_time=datetime.now(),
        series_volumes=[mock_dicom_volume.model_dump()],
    ).model_dump()

    mocks["retrieval"].execute_instruction.return_value = AgentResponse(
        instruction_id="inst1",
        agent_name="retrieval",
        status="SUCCESS",
        output_data={"retrieval_info": retrieval_info_data},
    )

    segmentation_result_data = ImageSegmentationResult(
        source_volume=mock_dicom_volume,
        segmentation_id="seg_456",
        regions_of_interest=[RegionOfInterest(roi_id="roi1", label="lesion", pixel_count=100)],
        segmentation_method="DL-Mock",
        processed_image_meta=ImageMetaData(original_path="dummy", image_format="nifti", dimensions=[1,2,3]),
    ).model_dump()
    mocks["image_processing"].execute_instruction.return_value = AgentResponse(
        instruction_id="inst2",
        agent_name="image_processing",
        status="SUCCESS",
        output_data={"segmentation_result": segmentation_result_data},
    )

    llm_analysis_result_data = LLMAnalysisResult(
        llm_output=LLMOutput(
            response_id="llm_resp1",
            model_name_used="mock_llm",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            choices=[LLMResponseChoice(index=0, text="Simulated LLM summary")],
        ),
        extracted_findings=[DiagnosticFinding(category="Anomaly", description="Mock finding", severity="MEDIUM", confidence_score=0.9)],
        summary="Mock LLM analysis summary.",
    ).model_dump()
    mocks["llm_inference"].execute_instruction.return_value = AgentResponse(
        instruction_id="inst3",
        agent_name="llm_inference",
        status="SUCCESS",
        output_data={"llm_analysis_result": llm_analysis_result_data},
    )

    mocks["semantic_indexing"].execute_instruction.return_value = AgentResponse(
        instruction_id="inst4",
        agent_name="semantic_indexing",
        status="SUCCESS",
        output_data={"status": "indexed"},
    )

    report_path = os.path.join(test_config.app.output_directory, "mock_report.json")
    mocks["reporting"].execute_instruction.return_value = AgentResponse(
        instruction_id="inst5",
        agent_name="reporting",
        status="SUCCESS",
        output_data={"report_path": report_path, "report_summary": {"overall_conclusion": "Mock final conclusion"}},
    )

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    dummy_report_data = DiagnosticReport(
        patient_info=patient_info,
        study_info=study_info,
        overall_conclusion="Mock final conclusion.",
        diagnostic_findings=[],
    ).model_dump_json()
    with open(report_path, "w") as f:
        f.write(dummy_report_data)

    task_definition = TaskDefinition(
        task_id="test_pipeline_id_1",
        task_name="Full Diagnostic Pipeline Test",
        sequence_of_instructions=[
            AgentInstruction(command="RETRIEVE_DICOM_STUDY", target_agent="retrieval", parameters={"patient_id": "test_patient"}),
            AgentInstruction(command="PERFORM_DL_SEGMENTATION", target_agent="image_processing", parameters={"dicom_volume": {"$from_context": "retrieval_info.series_volumes.0"}}),
            AgentInstruction(command="ANALYZE_WITH_LLM", target_agent="llm_inference", parameters={"prompt_template_name": "DiagnosticAssessment", "context_data": {"patient_info_summary": {"$from_context": "retrieval_info.patient_info"}, "image_analysis_summary": {"$from_context": "segmentation_result.regions_of_interest"}, "clinical_context": "No notes"}}),
            AgentInstruction(command="INDEX_DOCUMENT", target_agent="semantic_indexing", parameters={"doc_id": {"$from_context": "llm_analysis_result.analysis_id"}, "text_content": {"$from_context": "llm_analysis_result.summary"}}),
            AgentInstruction(command="GENERATE_DIAGNOSTIC_REPORT", target_agent="reporting", parameters={"patient_info": {"$from_context": "retrieval_info.patient_info"}, "study_info": {"$from_context": "retrieval_info.studies.0"}, "diagnostic_findings": {"$from_context": "llm_analysis_result.extracted_findings"}, "image_summaries_json": {"$from_context": "segmentation_result"}, "llm_summaries_json": {"$from_context": "llm_analysis_result"}}),
        ],
    )

    pipeline_status = await harmonizer.run_task_pipeline(task_definition)

    assert pipeline_status.overall_status == "COMPLETED_SUCCESS"
    assert len(pipeline_status.tasks_status) == 5
    for task_status in pipeline_status.tasks_status:
        assert task_status.status == "COMPLETED"

    mocks["retrieval"].execute_instruction.assert_awaited_once()
    mocks["image_processing"].execute_instruction.assert_awaited_once()
    mocks["llm_inference"].execute_instruction.assert_awaited_once()
    mocks["semantic_indexing"].execute_instruction.assert_awaited_once()
    mocks["reporting"].execute_instruction.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_task_pipeline_failure_halts(harmonizer_instance, mock_dicom_volume):
    harmonizer, mocks = harmonizer_instance
    retrieval_info_data = RetrievedDataInfo(
        retrieval_id="ret_123",
        data_source=DataSource(source_id="test", source_type="Local_Filesystem"),
        original_query="test_patient",
        retrieved_file_paths=[mock_dicom_volume.volume_path],
        total_files_retrieved=1,
        total_size_bytes=1000,
        retrieval_start_time=datetime.now(),
        retrieval_end_time=datetime.now(),
        series_volumes=[mock_dicom_volume.model_dump()],
    ).model_dump()

    mocks["retrieval"].execute_instruction.return_value = AgentResponse(
        instruction_id="inst1", agent_name="retrieval", status="SUCCESS", output_data={"retrieval_info": retrieval_info_data}
    )
    mocks["image_processing"].execute_instruction.return_value = AgentResponse(
        instruction_id="inst2", agent_name="image_processing", status="FAILED", error_message="Simulated image processing error."
    )

    task_definition = TaskDefinition(
        task_id="test_pipeline_id_2",
        task_name="Pipeline with Failure",
        sequence_of_instructions=[
            AgentInstruction(command="RETRIEVE_DICOM_STUDY", target_agent="retrieval", parameters={"patient_id": "test_patient"}),
            AgentInstruction(command="PERFORM_DL_SEGMENTATION", target_agent="image_processing", parameters={"dicom_volume": {"$from_context": "retrieval_info.series_volumes.0"}}),
            AgentInstruction(command="ANALYZE_WITH_LLM", target_agent="llm_inference", parameters={"prompt_template_name": "DiagnosticAssessment", "context_data": {}}),
        ],
    )

    pipeline_status = await harmonizer.run_task_pipeline(task_definition)

    assert pipeline_status.overall_status == "COMPLETED_WITH_ERRORS"
    assert "Simulated image processing error." in pipeline_status.overall_message
    assert len(pipeline_status.tasks_status) == 2
    assert pipeline_status.tasks_status[0].status == "COMPLETED"
    assert pipeline_status.tasks_status[1].status == "FAILED"

    mocks["retrieval"].execute_instruction.assert_awaited_once()
    mocks["image_processing"].execute_instruction.assert_awaited_once()
    mocks["llm_inference"].execute_instruction.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_not_found(harmonizer_instance):
    harmonizer, mocks = harmonizer_instance
    task_definition = TaskDefinition(
        task_id="test_pipeline_id_3",
        task_name="Pipeline Unknown Agent",
        sequence_of_instructions=[AgentInstruction(command="DO_SOMETHING", target_agent="NonExistentAgent", parameters={})],
    )
    pipeline_status = await harmonizer.run_task_pipeline(task_definition)

    assert pipeline_status.overall_status == "FAILED"
    assert "Target agent 'NonExistentAgent' not found" in pipeline_status.overall_message
    assert len(pipeline_status.tasks_status) == 1
    assert pipeline_status.tasks_status[0].status == "FAILED"
    assert pipeline_status.tasks_status[0].error_details["type"] == "AgentOrchestrationError"


@pytest.mark.asyncio
async def test_retrieval_agent_execute_instruction(test_config, mock_dicom_volume):
    agent = RetrievalAgent(test_config)
    with patch('msk_io.retrieval.dicom_stream_sniffer.DICOMStreamSniffer.discover_and_retrieve_studies', MagicMock()) as mock_discover:
        mock_discover.return_value = MagicMock(model_dump=lambda: {"mock": "retrieval_info", "series_volumes": [mock_dicom_volume.model_dump()]})
        instruction = AgentInstruction(command="RETRIEVE_DICOM_STUDY", parameters={"patient_id": "test_patient"}, target_agent="retrieval")
        response = await agent.execute_instruction(instruction, {})
        assert response.status == "SUCCESS"
        assert response.output_data == {"retrieval_info": {"mock": "retrieval_info", "series_volumes": [mock_dicom_volume.model_dump()]}}
        mock_discover.assert_called_once_with("test_patient", None)


@pytest.mark.asyncio
async def test_image_processing_agent_execute_instruction(test_config, mock_dicom_volume):
    agent = ImageProcessingAgent(test_config)
    with patch('msk_io.image_processing.dl_segmentor.DLSegmentor.segment_image_volume', MagicMock()) as mock_dl_seg:
        mock_dl_seg.return_value = MagicMock(model_dump=lambda: {"mock": "dl_segmentation_result"})
        instruction = AgentInstruction(command="PERFORM_DL_SEGMENTATION", parameters={"dicom_volume": mock_dicom_volume.model_dump()}, target_agent="image_processing")
        response = await agent.execute_instruction(instruction, {})
        assert response.status == "SUCCESS"
        assert response.output_data == {"segmentation_result": {"mock": "dl_segmentation_result"}}
        mock_dl_seg.assert_called_once()
        assert isinstance(mock_dl_seg.call_args.args[0], DICOMVolume)


@pytest.mark.asyncio
async def test_llm_inference_agent_execute_instruction(test_config):
    agent = LLMInferenceAgent(test_config)
    mock_llm_agent = AsyncMock()
    mock_llm_agent.analyze_data.return_value = MagicMock(model_dump=lambda: {"mock": "llm_analysis_result"})
    with patch('msk_io.inference.llm_agents.LLMAgentFactory.get_agent', return_value=mock_llm_agent), \
         patch('msk_io.inference.prompts.agent_prompts.get_prompt_template', return_value=MagicMock(format=lambda **kwargs: "formatted prompt")):
        instruction = AgentInstruction(command="ANALYZE_WITH_LLM", parameters={"agent_type": "mock_llm", "prompt_template_name": "MockTemplate", "context_data": {"test_key": "test_value"}}, target_agent="llm_inference")
        response = await agent.execute_instruction(instruction, {})
        assert response.status == "SUCCESS"
        assert response.output_data == {"llm_analysis_result": {"mock": "llm_analysis_result"}}
        mock_llm_agent.analyze_data.assert_awaited_once()


@pytest.mark.asyncio
async def test_semantic_indexing_agent_execute_instruction(test_config):
    agent = SemanticIndexingAgent(test_config)
    with patch('msk_io.indexer.semantic_indexer.SemanticIndexer.index_document', MagicMock()) as mock_index_doc, \
         patch('msk_io.indexer.semantic_indexer.SemanticIndexer.query_semantic_index', MagicMock()) as mock_query_index:
        instruction_index = AgentInstruction(command="INDEX_DOCUMENT", parameters={"doc_id": "d1", "text_content": "some text"}, target_agent="semantic_indexing")
        response_index = await agent.execute_instruction(instruction_index, {})
        assert response_index.status == "SUCCESS"
        assert response_index.output_data == {"status": "indexed", "doc_id": "d1"}
        mock_index_doc.assert_called_once_with("d1", "some text", None)
        mock_query_index.return_value = [{"result": "doc_found"}]
        instruction_query = AgentInstruction(command="QUERY_INDEX", parameters={"query_text": "query"}, target_agent="semantic_indexing")
        response_query = await agent.execute_instruction(instruction_query, {})
        assert response_query.status == "SUCCESS"
        assert response_query.output_data == {"query_results": [{"result": "doc_found"}]}
        mock_query_index.assert_called_once_with("query", 5)


@pytest.mark.asyncio
async def test_reporting_agent_execute_instruction(test_config, mock_dicom_volume):
    agent = ReportingAgent(test_config)
    mock_llm_agent_for_report = AsyncMock()
    mock_llm_agent_for_report.analyze_data.return_value = MagicMock(
        llm_output=MagicMock(choices=[MagicMock(text=json.dumps({"overall_conclusion": "LLM generated conclusion", "recommendations": ["Do more tests"]}))]),
        summary="LLM summary for report."
    )
    with patch('msk_io.inference.llm_agents.LLMAgentFactory.get_agent', return_value=mock_llm_agent_for_report), \
         patch('msk_io.inference.prompts.agent_prompts.get_prompt_template', return_value=MagicMock(format=lambda **kwargs: "formatted report prompt")):
        patient_info_dict = mock_dicom_volume.patient_info.model_dump()
        study_info_dict = mock_dicom_volume.study_info.model_dump()
        mock_img_analysis = ImageSegmentationResult(analyzed_volume=mock_dicom_volume, analysis_id="img_anal_1", qualitative_observations="img ok").model_dump()
        mock_llm_analysis = LLMAnalysisResult(llm_output=LLMOutput(response_id="llm_out_1", model_name_used="dummy", input_tokens=1, output_tokens=1, total_tokens=2, choices=[]), extracted_findings=[], summary="llm summary").model_dump()
        instruction = AgentInstruction(command="GENERATE_DIAGNOSTIC_REPORT", parameters={"patient_info": patient_info_dict, "study_info": study_info_dict, "diagnostic_findings": [], "image_summaries_json": [mock_img_analysis], "llm_summaries_json": [mock_llm_analysis]}, target_agent="reporting")
        response = await agent.execute_instruction(instruction, {})
        assert response.status == "SUCCESS"
        assert "report_path" in response.output_data
        assert "report_summary" in response.output_data
        assert response.output_data["report_summary"]["overall_conclusion"] == "LLM generated conclusion"
        mock_llm_agent_for_report.analyze_data.assert_awaited_once()
        assert os.path.exists(response.output_data["report_path"])
        with open(response.output_data["report_path"], 'r') as f:
            report_content = json.load(f)
            assert report_content["patient_info"]["patient_id"] == patient_info_dict["patient_id"]
            assert report_content["overall_conclusion"] == "LLM generated conclusion"

