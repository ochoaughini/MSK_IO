"""Run segmentation and mapping on a DICOM series."""
from __future__ import annotations

from pathlib import Path
import typer

from msk_io.api import PipelineRunner
from msk_io.config import PipelineSettings
from msk_io.storage.memory_vault import MemoryVault

app = typer.Typer(add_completion=False)


@app.command()
def analyze(path: str) -> None:
    settings = PipelineSettings(data_path=Path(path))
    result = PipelineRunner().run(settings, MemoryVault(settings.vault.path))
    typer.echo(result)


if __name__ == "__main__":  # pragma: no cover
    app()
