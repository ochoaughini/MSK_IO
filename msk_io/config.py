from pathlib import Path
from threading import Thread
from typing import Callable, Optional

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings


class PipelineSettings(BaseSettings):
    """Pipeline hyperparameters loaded from environment or file."""

    rules_path: Path = Field(..., description="Path to lattice rules JSON")
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    patch_size: int = Field(64, gt=0)
    stride: int = Field(32, gt=0)
    atlas_path: Optional[Path] = None
    verbose: bool = False

    model_config = dict(env_prefix="MSK_", extra="ignore")

    @field_validator("rules_path")
    @classmethod
    def check_rules(cls, v: Path) -> Path:
        if not v.exists():
            raise FileNotFoundError(v)
        return v


def start_settings_watcher(path: Path, callback: Callable[[PipelineSettings], None]) -> Thread:
    """Start a watchdog thread that reloads settings on file changes."""
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    class _Handler(FileSystemEventHandler):
        def on_modified(self, event):
            if Path(event.src_path) == path:
                try:
                    settings = PipelineSettings.model_validate_json(path.read_text())
                    callback(settings)
                except ValidationError as exc:
                    print(f"Config reload failed: {exc}")

    observer = Observer()
    observer.schedule(_Handler(), path.parent, recursive=False)
    thread = Thread(target=observer.start, daemon=True)
    thread.start()
    return thread
