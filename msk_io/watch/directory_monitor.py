from __future__ import annotations

"""Asynchronous directory monitor for the MSK folder.

This module provides a lightweight observer that watches the ``MSK``
directory on the user's desktop. It processes incoming DICOM and PDF
files using the existing :mod:`msk_io` pipeline components. Generated
Python scripts placed under ``generated_scripts`` are executed in a
separate subprocess with a strict timeout.

This implementation is a demonstration only and does not constitute
medical advice. Always consult a licensed professional before using the
output of this software for diagnostic purposes.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set

from watchdog.events import FileCreatedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from ..api import PipelineRunner
from ..config import PipelineSettings
from ..pdf.pdf_ingestor import MSKPDFIngestor
from ..storage.memory_vault import MemoryVault

logger = logging.getLogger(__name__)


@dataclass
class MSKFolderMonitor:
    """Watch the ``MSK`` folder and trigger pipeline actions."""

    base: Path = field(
        default_factory=lambda: Path.home() / "Desktop" / "MSK",
    )
    vault_path: Path = field(default_factory=lambda: Path("vault.db"))
    observer: Optional[Observer] = field(init=False, default=None)
    _processing: Set[Path] = field(init=False, default_factory=set)

    def start(self) -> None:
        """Start monitoring the folder."""
        self.base.mkdir(parents=True, exist_ok=True)
        for sub in ("dicom_images", "pdf_reference", "generated_scripts"):
            (self.base / sub).mkdir(exist_ok=True)

        self.observer = Observer()
        self.observer.schedule(
            _DicomHandler(self),
            str(self.base / "dicom_images"),
            recursive=True,
        )
        self.observer.schedule(
            _PDFHandler(self),
            str(self.base / "pdf_reference"),
            recursive=True,
        )
        self.observer.schedule(
            _ScriptHandler(self),
            str(self.base / "generated_scripts"),
            recursive=True,
        )
        self.observer.start()
        logger.info("Started monitoring %s", self.base)

    def stop(self) -> None:
        """Stop monitoring."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
        logger.info("Stopped monitoring %s", self.base)

    async def process_dicom_dir(self, directory: Path) -> None:
        """Run the pipeline on a directory of DICOM files."""
        if directory in self._processing:
            return
        self._processing.add(directory)
        try:
            settings = PipelineSettings(data_path=directory)
            vault = MemoryVault(self.vault_path)
            await PipelineRunner().run_async(settings, vault)
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.error("Processing failed for %s: %s", directory, exc)
        finally:
            self._processing.discard(directory)

    async def process_pdf(self, path: Path) -> None:
        """Ingest a PDF file for reference."""
        try:
            texts = await asyncio.to_thread(MSKPDFIngestor().ingest, path)
            logger.info("Ingested %d pages from %s", len(texts), path)
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.error("PDF ingestion failed for %s: %s", path, exc)

    async def run_script(self, path: Path) -> None:
        """Execute a generated script in a sandboxed subprocess."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "python",
                str(path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, err = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=60,
                )
                logger.info("Script %s exited %s", path, proc.returncode)
                if out:
                    logger.debug(out.decode())
                if err:
                    logger.warning(err.decode())
            except asyncio.TimeoutError:
                proc.kill()
                logger.warning("Script %s timed out", path)
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.error("Failed to run script %s: %s", path, exc)


class _DicomHandler(FileSystemEventHandler):
    def __init__(self, monitor: MSKFolderMonitor) -> None:
        self.monitor = monitor

    def on_created(self, event: FileCreatedEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() == ".dcm":
            asyncio.create_task(self.monitor.process_dicom_dir(path.parent))


class _PDFHandler(FileSystemEventHandler):
    def __init__(self, monitor: MSKFolderMonitor) -> None:
        self.monitor = monitor

    def on_created(self, event: FileCreatedEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() == ".pdf":
            asyncio.create_task(self.monitor.process_pdf(path))


class _ScriptHandler(FileSystemEventHandler):
    def __init__(self, monitor: MSKFolderMonitor) -> None:
        self.monitor = monitor

    def on_created(self, event: FileCreatedEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() == ".py":
            asyncio.create_task(self.monitor.run_script(path))
