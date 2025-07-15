from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4
from pydantic import Field
from msk_io.schema._pydantic_base import MSKIOBaseModel
from msk_io.schema.image_analysis import ImageAnalysisResult
from msk_io.schema.llm_output import LLMAnalysisResult, DiagnosticFinding
from msk_io.schema.retrieval_info import RetrievedDataInfo

class MultiModalInput(MSKIOBaseModel):
    text_data: Optional[str] = None
    image_analysis_results: List[ImageAnalysisResult] = []
    llm_analysis_results: List[LLMAnalysisResult] = []
    retrieved_data_info: Optional[RetrievedDataInfo] = None

class MultiModalSynthesisResult(MSKIOBaseModel):
    synthesis_id: str = Field(default_factory=lambda: str(uuid4()))
    input_data: MultiModalInput
    synthesized_conclusion: str
    supporting_findings: List[DiagnosticFinding] = []
    consistency_score: Optional[float] = None
    identified_discrepancies: List[str] = []
    synthesized_at: datetime = datetime.now()
    status: Literal["SUCCESS", "FAILURE", "PARTIAL_SUCCESS"] = "SUCCESS"
    errors: List[Dict[str, Any]] = []

    def add_finding(self, finding: DiagnosticFinding) -> None:
        self.supporting_findings.append(finding)
