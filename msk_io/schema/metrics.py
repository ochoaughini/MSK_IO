from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import uuid4
from pydantic import Field
from msk_io.schema._pydantic_base import MSKIOBaseModel

class Metric(MSKIOBaseModel):
    name: str
    value: float
    unit: Optional[str] = None
    description: Optional[str] = None

class EvaluationReport(MSKIOBaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid4()))
    evaluation_target: str
    evaluated_entity_id: Optional[str] = None
    timestamp: datetime = datetime.now()
    metrics: List[Metric] = []
    dataset_info: Dict[str, Any] = {}
    configuration_used: Dict[str, Any] = {}
    qualitative_observations: Optional[str] = None
    recommendations: List[str] = []
    status: Literal["PASSED", "FAILED", "IN_PROGRESS", "N/A"] = "N/A"
    errors: List[Dict[str, Any]] = []

    def add_metric(self, metric: Metric) -> None:
        self.metrics.append(metric)

    def add_metrics(self, metrics: List[Metric]) -> None:
        self.metrics.extend(metrics)
