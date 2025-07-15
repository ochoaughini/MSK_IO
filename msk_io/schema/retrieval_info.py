from typing import List, Literal, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4
from pydantic import Field

from msk_io.schema._pydantic_base import MSKIOBaseModel


class DataSource(MSKIOBaseModel):
    """Information about a data source from which medical data was retrieved."""

    source_id: str
    source_type: Literal["DICOM_PACS", "OHIF_Viewer", "Local_Filesystem", "Cloud_Storage"]
    endpoint_url: Optional[str] = None
    access_method: Optional[str] = None
    last_accessed: Optional[datetime] = None


class RetrievedDataInfo(MSKIOBaseModel):
    """Summary of data retrieved for a specific processing run."""

    retrieval_id: str = Field(default_factory=lambda: str(uuid4()))
    data_source: DataSource
    original_query: Optional[str] = None
    retrieved_file_paths: List[str] = Field(default_factory=list)
    total_files_retrieved: int
    total_size_bytes: Optional[int] = None
    retrieval_start_time: datetime = Field(default_factory=datetime.now)
    retrieval_end_time: datetime = Field(default_factory=datetime.now)
    status: Literal["SUCCESS", "FAILURE", "PARTIAL_SUCCESS"] = "SUCCESS"
    message: Optional[str] = None
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    series_volumes: List[Any] = Field(default_factory=list)

