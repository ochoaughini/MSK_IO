from __future__ import annotations

from pathlib import Path
import os
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoaderConfig(BaseModel):
    cache_lmdb: Optional[Path] = None
    interpolate_missing: bool = False

    @field_validator("cache_lmdb")
    @classmethod
    def validate_cache(cls, v: Optional[Path]) -> Optional[Path]:
        if v and v.exists() and not v.is_dir():
            raise ValueError("cache_lmdb must be a directory")
        return v


class ConverterConfig(BaseModel):
    embed_metadata: bool = True
    affine: Optional[List[List[float]]] = None


class SegmentorConfig(BaseModel):
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    model_path: Optional[Path] = None

    @field_validator("model_path")
    @classmethod
    def validate_model(cls, v: Optional[Path]) -> Optional[Path]:
        if v and not v.exists():
            raise FileNotFoundError(v)
        return v


class MapperConfig(BaseModel):
    mapping_file: Optional[Path] = None


class EmitterConfig(BaseModel):
    model: str = "default"


class LatticeConfig(BaseModel):
    rules_path: Path

    @field_validator("rules_path")
    @classmethod
    def validate_rules_path(cls, v: Path) -> Path:
        if not v.exists():
            raise FileNotFoundError(v)
        return v


class HarmonizerConfig(BaseModel):
    policy: str = "weighted"


class VaultConfig(BaseModel):
    path: Path = Path("vault.db")

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: Path) -> Path:
        if v.exists() and v.is_dir():
            raise ValueError("vault path must be a file")
        return v


class PipelineSettings(BaseSettings):
    loader: LoaderConfig = LoaderConfig()
    converter: ConverterConfig = ConverterConfig()
    segmentor: SegmentorConfig = SegmentorConfig()
    mapper: MapperConfig = MapperConfig()
    emitter: EmitterConfig = EmitterConfig()
    lattice: Optional[LatticeConfig] = None
    harmonizer: HarmonizerConfig = HarmonizerConfig()
    vault: VaultConfig = VaultConfig()

    data_path: Path = Path("./data")

    pdf_path: Optional[Path] = None
    ocr_enabled: bool = False
    vector_store_path: Path = Path("vector_store")
    log_level: str = "INFO"
    metrics_endpoint: Optional[str] = None
    vector_db_url: Optional[str] = None
    remote_url: Optional[str] = None
    auth_token: Optional[str] = None

    model_config = SettingsConfigDict(env_prefix="MSK_", extra="ignore")

    @field_validator("data_path")
    @classmethod
    def validate_data(cls, v: Path) -> Path:
        # When ``remote_url`` is used the local data path may not exist.
        if not v.exists():
            raise FileNotFoundError(v)
        return v

    def __init__(self, **data):
        rules_env = os.getenv("MSK_RULES_PATH")
        if rules_env and "lattice" not in data:
            data["lattice"] = {"rules_path": rules_env}
        thresh_env = os.getenv("MSK_THRESHOLD")
        if thresh_env and "segmentor" not in data:
            data["segmentor"] = {"threshold": float(thresh_env)}
        super().__init__(**data)

    @property
    def rules_path(self) -> Optional[Path]:
        if self.lattice:
            return self.lattice.rules_path
        return None

    @property
    def threshold(self) -> float:
        return self.segmentor.threshold


