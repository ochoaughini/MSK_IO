from __future__ import annotations

from pathlib import Path
from threading import Thread
from typing import Callable, Optional

from pydantic import ValidationError
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .config import PipelineSettings


class SettingsWatcher:
    """Watch a settings file for changes and hot-reload."""

    def __init__(self, path: Path, callback: Callable[[PipelineSettings], None]):
        self.path = path
        self.callback = callback
        self.thread: Optional[Thread] = None

    def start(self) -> Thread:
        class _Handler(FileSystemEventHandler):
            def on_modified(self, event):  # type: ignore[override]
                if Path(event.src_path) == self.path:
                    try:
                        settings = PipelineSettings.model_validate_json(self.path.read_text())
                        self.callback(settings)
                    except ValidationError as exc:  # pragma: no cover
                        print(f"Config reload failed: {exc}")

        observer = Observer()
        observer.schedule(_Handler(), self.path.parent, recursive=False)
        self.thread = Thread(target=observer.start, daemon=True)
        self.thread.start()
        return self.thread
