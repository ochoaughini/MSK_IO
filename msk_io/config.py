from __future__ import annotations

from pathlib import Path
import os
from threading import Thread
from typing import Callable, List, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoaderConfig(BaseModel):
    cache_lmdb: Optional[Path] = None
    interpolate_missing: bool = False


class ConverterConfig(BaseModel):
    embed_metadata: bool = True
    affine: Optional[List[List[float]]] = None


class SegmentorConfig(BaseModel):
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    model_path: Optional[Path] = None


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

    model_config = SettingsConfigDict(env_prefix="MSK_", extra="ignore")

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


class SettingsWatcher:
    def __init__(self, path: Path, callback: Callable[[PipelineSettings], None]):
        self.path = path
        self.callback = callback
        self.thread: Optional[Thread] = None

    def start(self) -> Thread:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        class _Handler(FileSystemEventHandler):
            def on_modified(self, event):  # type: ignore[override]
                if Path(event.src_path) == self.path:
                    try:
                        settings = PipelineSettings.model_validate_json(
                            self.path.read_text()
                        )
                        self.callback(settings)
                    except ValidationError as exc:  # pragma: no cover
                        print(f"Config reload failed: {exc}")

        observer = Observer()
        observer.schedule(_Handler(), self.path.parent, recursive=False)
        self.thread = Thread(target=observer.start, daemon=True)
        self.thread.start()
        return self.thread
