from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from .api import PipelineRunner
from .config import PipelineSettings, SettingsWatcher
from .pdf.pdf_ingestor import MSKPDFIngestor
from .storage.memory_vault import MemoryVault

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=True)


def _load_settings(path: Optional[Path]) -> PipelineSettings:
    if path and path.exists():
        return PipelineSettings.model_validate_json(path.read_text())
    return PipelineSettings()


@app.callback()
def main(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file"),
    log_level: str = typer.Option("INFO", help="Logging level"),
    metrics_endpoint: Optional[str] = typer.Option(None, "--metrics-endpoint"),
    vector_db_url: Optional[str] = typer.Option(None, "--vector-db-url"),
) -> None:
    settings = _load_settings(config)
    settings.log_level = log_level
    settings.metrics_endpoint = metrics_endpoint
    settings.vector_db_url = vector_db_url
    ctx.obj = settings
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, handlers=[RichHandler(rich_tracebacks=True)])


@app.command()
def run(ctx: typer.Context, pdf: Optional[Path] = typer.Option(None, "--pdf")) -> None:
    settings: PipelineSettings = ctx.obj
    if pdf:
        ingestor = MSKPDFIngestor()
        ingestor.ingest(pdf)
    vault = MemoryVault(settings.vault.path)
    runner = PipelineRunner()
    result = runner.run(settings, vault)
    console.print_json(data=result)


@app.command()
def ingest(ctx: typer.Context, pdf: Path) -> None:
    settings: PipelineSettings = ctx.obj
    ingestor = MSKPDFIngestor()
    texts = ingestor.ingest(pdf)
    console.print(f"Ingested {len(texts)} pages")


@app.command()
def index(ctx: typer.Context) -> None:
    settings: PipelineSettings = ctx.obj
    runner = PipelineRunner()
    runner.indexer.index_items(["dummy"])
    console.print("Index built")


@app.command()
def serve(ctx: typer.Context) -> None:
    console.print("Serving (stub)")


@app.command("dry-run")
def dry_run(ctx: typer.Context) -> None:
    settings: PipelineSettings = ctx.obj
    console.print(settings)


if __name__ == "__main__":  # pragma: no cover
    app()
