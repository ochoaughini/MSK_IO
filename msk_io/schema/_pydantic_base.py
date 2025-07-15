from pydantic import BaseModel, ConfigDict

class MSKIOBaseModel(BaseModel):
    """Base Pydantic model for all MSK-IO data structures."""
    model_config = ConfigDict(
        from_attributes=True,
        extra='forbid',
        arbitrary_types_allowed=False,
        populate_by_name=True,
        strict=True
    )
