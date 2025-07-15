from typing import Dict, List, Literal, Optional
from msk_io.schema._pydantic_base import MSKIOBaseModel

class PromptParameter(MSKIOBaseModel):
    name: str
    description: str
    is_required: bool = True
    default_value: Optional[str] = None
    example_value: Optional[str] = None
    data_type: Literal["string", "number", "boolean", "list", "json"] = "string"

class PromptTemplate(MSKIOBaseModel):
    template_name: str
    description: str
    template_string: str
    parameters: List[PromptParameter] = []
    expected_output_format: Optional[str] = None
    example_usage: Optional[str] = None

    def format(self, **kwargs: str) -> str:
        missing = [p.name for p in self.parameters if p.is_required and p.name not in kwargs]
        if missing:
            raise ValueError(f"Missing required prompt parameters: {', '.join(missing)}")
        return self.template_string.format(**kwargs)

class PromptSet(MSKIOBaseModel):
    set_name: str
    description: str
    prompts: List[PromptTemplate]
