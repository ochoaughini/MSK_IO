from datetime import datetime
from typing import List, Literal, Optional, Dict, Any
from uuid import uuid4
from pydantic import Field
from msk_io.schema._pydantic_base import MSKIOBaseModel

class DataSource(MSKIOBaseModel):
    source_id: str
    source_type: Literal["DICOM_PACS", "OHIF_Viewer", "Local_Filesystem", "Cloud_Storage"]
    endpoint_url: Optional[str] = None
    access_method: Optional[str] = None
    last_accessed: Optional[datetime] = None

class RetrievedDataInfo(MSKIOBaseModel):
    retrieval_id: str = Field(default_factory=lambda: str(uuid4()))
    data_source: DataSource
    original_query: Optional[str] = None
    retrieved_file_paths: List[str] = []
    total_files_retrieved: int
    total_size_bytes: Optional[int] = None
    retrieval_start_time: datetime
    retrieval_end_time: datetime
    status: Literal["SUCCESS", "FAILURE", "PARTIAL_SUCCESS"] = "SUCCESS"
    message: Optional[str] = None
    errors: List[Dict[str, Any]] = []
