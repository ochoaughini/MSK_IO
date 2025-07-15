from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4
from pydantic import Field
from msk_io.schema._pydantic_base import MSKIOBaseModel
from msk_io.schema.dicom_data import DICOMPatientInfo, DICOMStudyInfo, DICOMVolume
from msk_io.schema.image_analysis import ImageAnalysisResult
from msk_io.schema.llm_output import DiagnosticFinding, LLMAnalysisResult

class Attachment(MSKIOBaseModel):
    file_path: str
    description: Optional[str] = None
    mime_type: Optional[str] = None

class DiagnosticReport(MSKIOBaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid4()))
    patient_info: DICOMPatientInfo
    study_info: DICOMStudyInfo
    report_date: datetime = datetime.now()
    overall_conclusion: str
    diagnostic_findings: List[DiagnosticFinding] = []
    associated_volumes: List[DICOMVolume] = []
    image_analysis_summaries: List[ImageAnalysisResult] = []
    llm_analysis_summaries: List[LLMAnalysisResult] = []
    recommendations: List[str] = []
    status: Literal["FINAL", "PRELIMINARY", "PENDING_REVIEW", "ERROR"] = "PRELIMINARY"
    attachments: List[Attachment] = []
    reviewer_notes: Optional[str] = None

    def add_finding(self, finding: DiagnosticFinding) -> None:
        self.diagnostic_findings.append(finding)

    def add_volume(self, volume: DICOMVolume) -> None:
        self.associated_volumes.append(volume)

    def add_image_analysis(self, analysis: ImageAnalysisResult) -> None:
        self.image_analysis_summaries.append(analysis)

    def add_llm_analysis(self, llm_analysis: LLMAnalysisResult) -> None:
        self.llm_analysis_summaries.append(llm_analysis)

    def add_recommendation(self, recommendation: str) -> None:
        self.recommendations.append(recommendation)

    def add_attachment(self, attachment: Attachment) -> None:
        self.attachments.append(attachment)
