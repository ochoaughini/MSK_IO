from typing import Any, Dict, List, Literal, Optional
from datetime import datetime
from uuid import uuid4
from pydantic import Field

from msk_io.schema._pydantic_base import MSKIOBaseModel


class AgentInstruction(MSKIOBaseModel):
    """A specific instruction for an agent to perform."""

    instruction_id: str = Field(default_factory=lambda: str(uuid4()))
    command: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    target_agent: Optional[str] = None
    priority: int = Field(0, ge=0, description="Higher value means higher priority.")


class AgentResponse(MSKIOBaseModel):
    """The response from an agent after executing an instruction."""

    response_id: str = Field(default_factory=lambda: str(uuid4()))
    instruction_id: str
    agent_name: str
    status: Literal["SUCCESS", "FAILED", "PENDING"]
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class TaskDefinition(MSKIOBaseModel):
    """Defines a complex task composed of multiple agent instructions."""

    task_id: str = Field(default_factory=lambda: str(uuid4()))
    task_name: str
    description: Optional[str] = None
    required_inputs: List[str] = Field(default_factory=list)
    output_type: Optional[str] = None
    sequence_of_instructions: List[AgentInstruction] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)

