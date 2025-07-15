from datetime import datetime
from typing import List, Optional, Dict, Any, Literal
from uuid import uuid4
from pydantic import Field
from msk_io.schema._pydantic_base import MSKIOBaseModel

class ServiceHealth(MSKIOBaseModel):
    service_name: str
    is_healthy: bool
    status_message: str
    last_checked: datetime = datetime.now()
    details: Optional[Dict[str, Any]] = None

class SystemHealthReport(MSKIOBaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = datetime.now()
    overall_status: Literal["OPERATIONAL", "DEGRADED", "OUTAGE"]
    message: Optional[str] = None
    service_statuses: List[ServiceHealth] = []

class RuntimeMetrics(MSKIOBaseModel):
    operation_name: str
    duration_seconds: float
    cpu_usage_percent: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    disk_io_mb_per_s: Optional[float] = None
    network_io_mb_per_s: Optional[float] = None
