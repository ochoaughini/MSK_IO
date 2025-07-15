from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from uuid import uuid4
from pydantic import Field

from msk_io.schema._pydantic_base import MSKIOBaseModel


class ServiceHealth(MSKIOBaseModel):
    """Provides health status of an internal service or component."""

    service_name: str
    is_healthy: bool
    status_message: str
    last_checked: datetime = Field(default_factory=datetime.now)
    details: Optional[Dict[str, Any]] = None


class SystemHealthReport(MSKIOBaseModel):
    """Overall system health report for the MSK-IO pipeline."""

    report_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    overall_status: Literal["OPERATIONAL", "DEGRADED", "OUTAGE"]
    message: Optional[str] = None
    service_statuses: List[ServiceHealth] = Field(default_factory=list)


class RuntimeMetrics(MSKIOBaseModel):
    """Runtime performance metrics for a specific operation."""

    operation_name: str
    duration_seconds: float
    cpu_usage_percent: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    disk_io_mb_per_s: Optional[float] = None
    network_io_mb_per_s: Optional[float] = None

