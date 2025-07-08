from pathlib import Path
from pydantic import BaseModel, Field, validator


class PipelineConfig(BaseModel):
    rules_path: Path = Field(..., description="Path to lattice rules JSON")
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    verbose: bool = False

    @validator("rules_path")
    def check_rules(cls, v: Path) -> Path:
        if not v.exists():
            raise FileNotFoundError(v)
        return v
