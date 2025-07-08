from __future__ import annotations

from pathlib import Path
from threading import Thread
from typing import Callable, Optional, List
import os

from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings


class LoaderConfig(BaseModel):
    """Configuration for DICOM loading."""

    cache_lmdb: Optional[Path] = None
    interpolate_missing: bool = False


class ConverterConfig(BaseModel):
    """Settings for NIfTI conversion."""

    embed_metadata: bool = True
    affine: Optional[List[List[float]]] = None


class SegmentorConfig(BaseModel):
    """Segmentation options."""

    threshold: float = Field(0.5, ge=0.0, le=1.0)
    model_path: Optional[Path] = None


class MapperConfig(BaseModel):
    mapping_file: Optional[Path] = None


class EmitterConfig(BaseModel):
    model: str = "default"


class LatticeConfig(BaseModel):
    rules_path: Path = Field(..., alias="rules_path")

    @field_validator("rules_path")
    @classmethod
    def check_rules(cls, v: Path) -> Path:
        if not v.exists():
            raise FileNotFoundError(v)
        return v


class HarmonizerConfig(BaseModel):
    policy: str = "weighted"


class VaultConfig(BaseModel):
    path: Path


class PipelineConfig(BaseSettings):
    """Root settings object for the entire pipeline."""

    loader: LoaderConfig = LoaderConfig()
    converter: ConverterConfig = ConverterConfig()
    segmentor: SegmentorConfig = SegmentorConfig()
    mapper: MapperConfig = MapperConfig()
    emitter: EmitterConfig = EmitterConfig()
    lattice: Optional[LatticeConfig] = None
    harmonizer: HarmonizerConfig = HarmonizerConfig()
    vault: Optional[VaultConfig] = None

    verbose: bool = False
    dry_run: bool = False
    debug: bool = False

    model_config = dict(env_prefix="MSK_", extra="ignore", populate_by_name=True)

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


# Backwards compatibility
PipelineSettings = PipelineConfig


def start_settings_watcher(
    path: Path, callback: Callable[[PipelineConfig], None]
) -> Thread:
    """Start a watchdog thread that reloads settings on file changes."""
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    class _Handler(FileSystemEventHandler):
        def on_modified(self, event):
            if Path(event.src_path) == path:
                try:
                    settings = PipelineConfig.model_validate_json(path.read_text())
                    callback(settings)
                except ValidationError as exc:
                    print(f"Config reload failed: {exc}")

    observer = Observer()
    observer.schedule(_Handler(), path.parent, recursive=False)
    thread = Thread(target=observer.start, daemon=True)
    thread.start()
    return thread
