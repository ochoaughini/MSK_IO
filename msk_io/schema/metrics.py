from typing import Dict, Any, List, Optional, Literal
from datetime import datetime
from uuid import uuid4
from pydantic import Field

from msk_io.schema._pydantic_base import MSKIOBaseModel


class Metric(MSKIOBaseModel):
    """A single evaluation metric."""

    name: str
    value: float
    unit: Optional[str] = None
    description: Optional[str] = None


class EvaluationReport(MSKIOBaseModel):
    """A comprehensive report of an evaluation run."""

    report_id: str = Field(default_factory=lambda: str(uuid4()))
    evaluation_target: str
    evaluated_entity_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metrics: List[Metric] = Field(default_factory=list)
    dataset_info: Dict[str, Any] = Field(default_factory=dict)
    configuration_used: Dict[str, Any] = Field(default_factory=dict)
    qualitative_observations: Optional[str] = None
    recommendations: List[str] = Field(default_factory=list)
    status: Literal["PASSED", "FAILED", "IN_PROGRESS", "N/A"] = "N/A"
    errors: List[Dict[str, Any]] = Field(default_factory=list)

    def add_metric(self, metric: Metric) -> None:
        self.metrics.append(metric)

    def add_metrics(self, metrics: List[Metric]) -> None:
        self.metrics.extend(metrics)

