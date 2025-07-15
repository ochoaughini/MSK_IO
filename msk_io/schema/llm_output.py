from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4
from pydantic import Field
from msk_io.schema._pydantic_base import MSKIOBaseModel

class LLMInput(MSKIOBaseModel):
    prompt_text: str
    context_data: Dict[str, Any] = {}
    model_name: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

class LLMResponseChoice(MSKIOBaseModel):
    text: str
    index: int
    logprobs: Optional[float] = None
    finish_reason: Optional[str] = None

class LLMOutput(MSKIOBaseModel):
    response_id: str
    model_name_used: str
    timestamp: datetime = datetime.now()
    input_tokens: int
    output_tokens: int
    total_tokens: int
    choices: List[LLMResponseChoice]
    raw_response: Dict[str, Any] = {}

class DiagnosticFinding(MSKIOBaseModel):
    finding_id: str = Field(default_factory=lambda: str(uuid4()))
    category: str
    description: str
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW", "NORMAL"]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    associated_roi_ids: List[str] = []
    supporting_evidence_texts: List[str] = []
    recommended_action: Optional[str] = None

class LLMAnalysisResult(MSKIOBaseModel):
    analysis_id: str = Field(default_factory=lambda: str(uuid4()))
    llm_output: LLMOutput
    extracted_findings: List[DiagnosticFinding]
    summary: str
    status: Literal["SUCCESS", "FAILURE", "PARTIAL_SUCCESS"] = "SUCCESS"
    errors: List[Dict[str, Any]] = []

    def add_finding(self, finding: DiagnosticFinding) -> None:
        self.extracted_findings.append(finding)
