from uuid import uuid4
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import Field
from msk_io.schema._pydantic_base import MSKIOBaseModel

class AgentInstruction(MSKIOBaseModel):
    instruction_id: str = Field(default_factory=lambda: str(uuid4()))
    command: str
    parameters: Dict[str, Any] = {}
    target_agent: Optional[str] = None
    priority: int = Field(0, ge=0)

class AgentResponse(MSKIOBaseModel):
    response_id: str = Field(default_factory=lambda: str(uuid4()))
    instruction_id: str
    agent_name: str
    status: Literal["SUCCESS", "FAILED", "PENDING"]
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    timestamp: datetime = datetime.now()

class TaskDefinition(MSKIOBaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    task_name: str
    description: Optional[str] = None
    required_inputs: List[str] = []
    output_type: Optional[str] = None
    sequence_of_instructions: List[AgentInstruction] = []
    dependencies: List[str] = []
