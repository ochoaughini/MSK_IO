from typing import Any, Dict, List, Literal, Optional
from datetime import datetime
from uuid import uuid4
from pydantic import Field

from msk_io.schema._pydantic_base import MSKIOBaseModel
from msk_io.schema.image_analysis import ImageAnalysisResult
from msk_io.schema.llm_output import LLMAnalysisResult, DiagnosticFinding
from msk_io.schema.retrieval_info import RetrievedDataInfo
from msk_io.schema.dicom_data import DICOMVolume


class MultiModalInput(MSKIOBaseModel):
    """Aggregated input from various modalities for synthesis."""

    text_data: Optional[str] = None
    image_analysis_results: List[ImageAnalysisResult] = Field(default_factory=list)
    llm_analysis_results: List[LLMAnalysisResult] = Field(default_factory=list)
    retrieved_data_info: Optional[RetrievedDataInfo] = None


class MultiModalSynthesisResult(MSKIOBaseModel):
    """The result of synthesizing information from multiple modalities."""

    synthesis_id: str = Field(default_factory=lambda: str(uuid4()))
    input_data: MultiModalInput
    synthesized_conclusion: str
    supporting_findings: List[DiagnosticFinding] = Field(default_factory=list)
    consistency_score: Optional[float] = None
    identified_discrepancies: List[str] = Field(default_factory=list)
    synthesized_at: datetime = Field(default_factory=datetime.now)
    status: Literal["SUCCESS", "FAILURE", "PARTIAL_SUCCESS"] = "SUCCESS"
    errors: List[Dict[str, Any]] = Field(default_factory=list)

    def add_finding(self, finding: DiagnosticFinding) -> None:
        self.supporting_findings.append(finding)

