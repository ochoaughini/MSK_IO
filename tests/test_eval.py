import pytest
from datetime import datetime, date

from msk_io.eval.evaluator import Evaluator
from msk_io.schema.metrics import EvaluationReport, Metric
from msk_io.schema.image_analysis import ImageSegmentationResult, RegionOfInterest, ImageMetaData
from msk_io.schema.llm_output import LLMAnalysisResult, LLMOutput, LLMResponseChoice, DiagnosticFinding
from msk_io.schema.dicom_data import DICOMVolume, DICOMPatientInfo, DICOMStudyInfo, DICOMSeriesInfo
from msk_io.errors import DataValidationError


@pytest.fixture
def evaluator_instance(test_config):
    return Evaluator(test_config)


@pytest.fixture
def mock_segmentation_result(mock_dicom_volume) -> ImageSegmentationResult:
    return ImageSegmentationResult(
        source_volume=mock_dicom_volume,
        segmentation_id="seg_test_id",
        regions_of_interest=[
            RegionOfInterest(roi_id="roi_1", label="Tumor", pixel_count=1000),
            RegionOfInterest(roi_id="roi_2", label="Organ", pixel_count=5000),
        ],
        segmentation_method="DL-UNet",
        processed_image_meta=ImageMetaData(
            original_path=mock_dicom_volume.volume_path,
            image_format="nifti",
            dimensions=[100, 100, 100],
        ),
    )


@pytest.fixture
def mock_llm_analysis_result() -> LLMAnalysisResult:
    return LLMAnalysisResult(
        llm_output=LLMOutput(
            response_id="llm_resp_test",
            model_name_used="gpt-4o",
            input_tokens=50,
            output_tokens=100,
            total_tokens=150,
            choices=[LLMResponseChoice(index=0, text="Analysis complete. Finding: Anomaly detected.")],
        ),
        extracted_findings=[
            DiagnosticFinding(category="Anomaly", description="Lesion detected.", severity="MEDIUM", confidence_score=0.8),
            DiagnosticFinding(category="Pathology", description="Inflammation suspected.", severity="LOW", confidence_score=0.6),
        ],
        summary="Summary of findings from LLM.",
    )


@pytest.mark.asyncio
async def test_evaluate_segmentation_success(evaluator_instance, mock_segmentation_result):
    ground_truth = [{"roi_id": "gt_roi_1", "mask_path": "/path/to/gt_mask.nii.gz", "label": "Tumor"}]
    report = await evaluator_instance.evaluate_segmentation(ground_truth, mock_segmentation_result)
    assert isinstance(report, EvaluationReport)
    assert report.evaluation_target == "Segmentation Model"
    assert report.evaluated_entity_id == "seg_test_id"
    assert report.status == "PASSED"
    assert any(m.name == "Dice Coefficient (Simulated)" for m in report.metrics)


@pytest.mark.asyncio
async def test_evaluate_segmentation_no_ground_truth(evaluator_instance, mock_segmentation_result):
    with pytest.raises(DataValidationError):
        await evaluator_instance.evaluate_segmentation([], mock_segmentation_result)


@pytest.mark.asyncio
async def test_evaluate_llm_analysis_success(evaluator_instance, mock_llm_analysis_result):
    ground_truth = [DiagnosticFinding(category="Anomaly", description="Lesion detected.", severity="MEDIUM", confidence_score=0.9)]
    report = await evaluator_instance.evaluate_llm_analysis(ground_truth, mock_llm_analysis_result)
    assert isinstance(report, EvaluationReport)
    assert report.evaluation_target == "LLM Inference Agent"
    assert report.status in {"PASSED", "FAILED"}


@pytest.mark.asyncio
async def test_evaluate_llm_analysis_no_ground_truth(evaluator_instance, mock_llm_analysis_result):
    with pytest.raises(DataValidationError):
        await evaluator_instance.evaluate_llm_analysis([], mock_llm_analysis_result)

